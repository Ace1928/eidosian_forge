import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_closer_than(self):
    """Test words_closer_than returns expected value for distinct and identical nodes."""
    self.assertEqual(self.vectors.closer_than('war', 'war'), [])
    expected = set(['conflict', 'administration'])
    self.assertEqual(set(self.vectors.closer_than('war', 'terrorism')), expected)