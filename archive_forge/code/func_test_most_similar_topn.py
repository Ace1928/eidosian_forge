import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_most_similar_topn(self):
    """Test most_similar returns correct results when `topn` is specified."""
    self.assertEqual(len(self.vectors.most_similar('war', topn=5)), 5)
    self.assertEqual(len(self.vectors.most_similar('war', topn=10)), 10)
    predicted = self.vectors.most_similar('war', topn=None)
    self.assertEqual(len(predicted), len(self.vectors))
    predicted = self.vectors.most_similar('war', topn=0)
    self.assertEqual(len(predicted), 0)
    predicted = self.vectors.most_similar('war', topn=np.uint8(0))
    self.assertEqual(len(predicted), 0)