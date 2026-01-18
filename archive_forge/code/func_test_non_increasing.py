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
def test_non_increasing(self):
    """ Check that similarities are non-increasing when `num_best` is not `None`."""
    index = self.cls(CORPUS, self.similarity_matrix, num_best=5)
    query = DICTIONARY.doc2bow(TEXTS[0])
    sims = index[query]
    sims2 = numpy.asarray(sims)[:, 1]
    cond = sum(numpy.diff(sims2) <= 0) == len(sims2) - 1
    self.assertTrue(cond)