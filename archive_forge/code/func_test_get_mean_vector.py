import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_get_mean_vector(self):
    """Test get_mean_vector returns expected results."""
    keys = ['conflict', 'administration', 'terrorism', 'call', 'an out-of-vocabulary word']
    weights = [1, 2, 3, 1, 2]
    expected_result_1 = np.array([0.02000151, -0.12685453, 0.09196121, 0.25514853, 0.25740655, -0.11134843, -0.0502661, -0.19278568, -0.83346179, -0.12068878], dtype=np.float32)
    expected_result_2 = np.array([-0.0145228, -0.11530358, 0.1169825, 0.22537769, 0.29353586, -0.10458107, -0.05272481, -0.17547795, -0.84245106, -0.10356515], dtype=np.float32)
    expected_result_3 = np.array([0.01343237, -0.47651053, 0.45645328, 0.98304356, 1.1840123, -0.51647933, -0.25308795, -0.77931081, -3.55954733, -0.55429711], dtype=np.float32)
    self.assertTrue(np.allclose(self.vectors.get_mean_vector(keys), expected_result_1))
    self.assertTrue(np.allclose(self.vectors.get_mean_vector(keys, weights), expected_result_2))
    self.assertTrue(np.allclose(self.vectors.get_mean_vector(keys, pre_normalize=False), expected_result_3))