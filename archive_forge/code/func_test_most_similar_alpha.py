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
def test_most_similar_alpha(self):
    index = LevenshteinSimilarityIndex(self.dictionary, alpha=1.0)
    first_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
    index = LevenshteinSimilarityIndex(self.dictionary, alpha=2.0)
    second_similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=10)])
    self.assertTrue(numpy.allclose(2.0 * first_similarities, second_similarities))