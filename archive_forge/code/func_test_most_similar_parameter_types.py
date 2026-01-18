import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_most_similar_parameter_types(self):
    """Are the positive/negative parameter types are getting interpreted correctly?"""
    partial = functools.partial(self.vectors.most_similar, topn=5)
    position = partial('war', 'peace')
    position_list = partial(['war'], ['peace'])
    keyword = partial(positive='war', negative='peace')
    keyword_list = partial(positive=['war'], negative=['peace'])
    assert position == position_list
    assert position == keyword
    assert position == keyword_list