import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_load_model_and_vocab_file_ignore(self):
    """Test loading model and voacab files which have decoding errors: ignore mode"""
    model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, fvocab=self.vocab_path, binary=False, unicode_errors='ignore')
    self.assertEqual(model.get_vecattr(u'ありがとう', 'count'), 123)
    self.assertEqual(model.get_vecattr(u'どういたしまして', 'count'), 789)
    self.assertEqual(model.key_to_index[u'ありがとう'], 0)
    self.assertEqual(model.key_to_index[u'どういたしまして'], 1)
    self.assertTrue(np.array_equal(model.get_vector(u'ありがとう'), np.array([0.6, 0.6, 0.6], dtype=np.float32)))
    self.assertTrue(np.array_equal(model.get_vector(u'どういたしまして'), np.array([0.1, 0.2, 0.3], dtype=np.float32)))