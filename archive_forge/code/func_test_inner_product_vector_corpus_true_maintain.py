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
def test_inner_product_vector_corpus_true_maintain(self):
    """Test the inner product between a vector and a corpus with the (True, 'maintain') normalization."""
    expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
    expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
    expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
    expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
    expected_result = numpy.full((1, 2), expected_result)
    result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, 'maintain'))
    self.assertTrue(isinstance(result, numpy.ndarray))
    self.assertTrue(numpy.allclose(expected_result, result))