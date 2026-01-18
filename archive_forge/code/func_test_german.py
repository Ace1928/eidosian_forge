import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
def test_german(self):
    stemmer_german = SnowballStemmer('german')
    stemmer_german2 = SnowballStemmer('german', ignore_stopwords=True)
    assert stemmer_german.stem('Schränke') == 'schrank'
    assert stemmer_german2.stem('Schränke') == 'schrank'
    assert stemmer_german.stem('keinen') == 'kein'
    assert stemmer_german2.stem('keinen') == 'keinen'