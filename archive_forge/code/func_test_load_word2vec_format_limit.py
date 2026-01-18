import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_load_word2vec_format_limit(self):
    w2v_dict = {'abc': [1, 2, 3], 'cde': [4, 5, 6], 'def': [7, 8, 9]}
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)
    w2v_dict = {'abc': [1, 2, 3], 'cde': [4, 5, 6], 'def': [7, 8, 9]}
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)
    w2v_dict = {'abc': [1, 2, 3], 'cdefg': [4, 5, 6], 'd': [7, 8, 9]}
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)
    w2v_dict = {'abc': [1, 2, 3], 'cdefg': [4, 5, 6], 'd': [7, 8, 9]}
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
    self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)