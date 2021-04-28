import sklearn
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
clf = MultinomialNB()
cv = CountVectorizer()

app = Flask(__name__)


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html")


@app.errorhandler(sklearn.exceptions.NotFittedError)
def all_exception_handler(e):
    return render_template("exception.html")


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/modelbuilding")
def modelbuilding():
    # Read dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    # Features and Labels
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    x = df['v2']
    y = df['label']

    # Preprocessing data
    # remove -ing
    snow = SnowballStemmer('english')  # Create a new instance of a language specific subclass.
    text = []
    for message in df['v2']:
        message = message.lower().strip()  # convert to lowercase
        message = nltk.re.sub(r"[^a-zA-z0-9]", " ", message)
        message = nltk.re.sub(r'\d+', '', message)
        words = [snow.stem(word)
                 for word in message.split()
                 if word not in stopwords.words('english')]
        text.append(' '.join(words))

    # Feature Extract With CountVectorizer
    x = cv.fit_transform(text).toarray()  # Fit the Data
    x_train, x_test, y_train, y_test = train_test_split(x, y)  # data splitting

    # Naive Bayes Classifier
    clf.fit(x_train, y_train)  # Fit Naive Bayes classifier according to x, y
    # Evaluation
    acc = int(round(clf.score(x_test, y_test)*100))  # Returns the mean accuracy on the given test data and labels.
    print('Accuracy: ', acc)

    return render_template("index.html", acc=acc, status='Success')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template("result.html", predict=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
