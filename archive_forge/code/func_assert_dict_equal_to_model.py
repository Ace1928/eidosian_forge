import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def assert_dict_equal_to_model(self, d, m):
    self.assertEqual(len(d), len(m))
    for word in d.keys():
        self.assertSequenceEqual(list(d[word]), list(m[word]))