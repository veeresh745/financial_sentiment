from flask import Flask
from flask import request

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
import nltk

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


df = pd.read_csv("fdata.csv")
df.head()
df.describe()
nltk.download("stopwords")


def clean_text(text):
    # create a list of stop words
    stop_words = [stopwords.words("english")]

    no_punc = [word for word in text if word not in string.punctuation]

    no_punc_str = "".join(no_punc)

    return "".join([word for word in no_punc_str if word.lower() not in stop_words])


df["cleaned"] = df["Sentence"].apply(clean_text)
df.head()

df["mapped_sentiments"] = np.where(
    df["Sentiment"] == "positive", 1, np.where(df["Sentiment"] == "negative", -1, 0)
)

df.head()

df = df.drop(["Sentence", "Sentiment"], axis=1)

df.head()

X = df["cleaned"]

y = df["mapped_sentiments"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

count_vec = CountVectorizer()
# Fit on train set
count_vec.fit(X_train)
# Transform the train and test sets

X_train_vec = count_vec.transform(X_train)
X_test_vec = count_vec.transform(X_test)


tfidf = TfidfTransformer()

tfidf.fit(X_train_vec)
tfidf.transform(X_train_vec)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)

accuracy_score(y_test, pred)

# testnews = ["Banks tumble as SVB ignites capitalization fears"]
# # clean_test_news = clean_text(testnews)
# clean_test_news1 = count_vec.transform(testnews)
# pred1 = model.predict(clean_test_news1)


@app.route("/", methods=["GET"])
def test():
    return {"status": "ok"}


@app.route("/predict", methods=["GET"])
@cross_origin()
def predict():
    input = request.args.get("q")
    print("\n\ninput: {0}\n\n".format(input))
    if len(input) == 0:
        return {"message": "query provided"}

    # predict
    clean_input = count_vec.transform([input])
    print("clean input: ", clean_input)
    pred1 = model.predict(clean_input)
    print("pred 1: ", pred1)
    return {"score": str(pred1[0])}
