import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_most_similar(self):
    """Test most_similar returns expected results."""
    expected = ['conflict', 'administration', 'terrorism', 'call', 'israel']
    predicted = [result[0] for result in self.vectors.most_similar('war', topn=5)]
    self.assertEqual(expected, predicted)