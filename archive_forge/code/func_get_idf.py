from collections import defaultdict
import math
import unittest
from gensim.models.bm25model import BM25ABC
from gensim.models import OkapiBM25Model, LuceneBM25Model, AtireBM25Model
from gensim.corpora import Dictionary
def get_idf(word):
    frequency = sum(map(lambda document: word in document, self.documents))
    return math.log(len(self.documents) / frequency)