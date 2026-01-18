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
def test_maintain_sparsity_with_num_best(self):
    """Tests that sparsity is correctly maintained when maintain_sparsity=True and num_best is not None"""
    num_features = len(DICTIONARY)
    index = self.cls(CORPUS, num_features=num_features, maintain_sparsity=False, num_best=3)
    dense_topn_sims = index[CORPUS]
    index = self.cls(CORPUS, num_features=num_features, maintain_sparsity=True, num_best=3)
    scipy_topn_sims = index[CORPUS]
    self.assertFalse(scipy.sparse.issparse(dense_topn_sims))
    self.assertTrue(scipy.sparse.issparse(scipy_topn_sims))
    self.assertEqual(dense_topn_sims, [matutils.scipy2sparse(v) for v in scipy_topn_sims])