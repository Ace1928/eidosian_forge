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
def test_encapsulation(self):
    """Test the matrix encapsulation."""
    expected_matrix = numpy.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    matrix = SparseTermSimilarityMatrix(scipy.sparse.csc_matrix(expected_matrix)).matrix
    self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))
    self.assertTrue(numpy.all(matrix.todense() == expected_matrix))
    matrix = SparseTermSimilarityMatrix(scipy.sparse.csr_matrix(expected_matrix)).matrix
    self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))
    self.assertTrue(numpy.all(matrix.todense() == expected_matrix))