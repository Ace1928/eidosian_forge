import logging
import unittest
import math
import os
import numpy
import scipy
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim import matutils, similarities
from gensim.models import Word2Vec, FastText
from gensim.test.utils import (
from gensim.similarities import UniformTermSimilarityIndex
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import _nlargest
from gensim.similarities.fastss import editdist
def test_empty_query(self):
    index = self.factoryMethod()
    if isinstance(index, similarities.WmdSimilarity) and (not POT_EXT):
        self.skipTest('POT not installed')
    query = []
    try:
        sims = index[query]
        self.assertTrue(sims is not None)
    except IndexError:
        self.assertTrue(False)