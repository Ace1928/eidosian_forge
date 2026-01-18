import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_load_model_and_vocab_file_strict(self):
    """Test loading model and voacab files which have decoding errors: strict mode"""
    with self.assertRaises(UnicodeDecodeError):
        gensim.models.KeyedVectors.load_word2vec_format(self.model_path, fvocab=self.vocab_path, binary=False, unicode_errors='strict')