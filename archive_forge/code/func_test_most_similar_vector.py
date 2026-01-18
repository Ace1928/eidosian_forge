import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_most_similar_vector(self):
    """Can we pass vectors to most_similar directly?"""
    positive = self.vectors.vectors[0:5]
    most_similar = self.vectors.most_similar(positive=positive)
    assert most_similar is not None