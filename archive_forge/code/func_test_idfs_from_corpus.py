from collections import defaultdict
import math
import unittest
from gensim.models.bm25model import BM25ABC
from gensim.models import OkapiBM25Model, LuceneBM25Model, AtireBM25Model
from gensim.corpora import Dictionary
def test_idfs_from_corpus(self):
    corpus = list(map(self.dictionary.doc2bow, self.documents))
    model = AtireBM25Model(corpus=corpus, k1=self.k1, b=self.b)
    actual_dog_idf = model.idfs[self.dictionary.token2id['dog']]
    actual_cat_idf = model.idfs[self.dictionary.token2id['cat']]
    actual_mouse_idf = model.idfs[self.dictionary.token2id['mouse']]
    actual_lion_idf = model.idfs[self.dictionary.token2id['lion']]
    self.assertAlmostEqual(self.expected_dog_idf, actual_dog_idf)
    self.assertAlmostEqual(self.expected_cat_idf, actual_cat_idf)
    self.assertAlmostEqual(self.expected_mouse_idf, actual_mouse_idf)
    self.assertAlmostEqual(self.expected_lion_idf, actual_lion_idf)