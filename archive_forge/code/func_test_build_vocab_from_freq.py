import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_build_vocab_from_freq(self):
    """Test that the algorithm is able to build vocabulary from given
        frequency table"""
    freq_dict = {'minors': 2, 'graph': 3, 'system': 4, 'trees': 3, 'eps': 2, 'computer': 2, 'survey': 2, 'user': 3, 'human': 2, 'time': 2, 'interface': 2, 'response': 2}
    freq_dict_orig = freq_dict.copy()
    model_hs = word2vec.Word2Vec(vector_size=10, min_count=0, seed=42, hs=1, negative=0)
    model_neg = word2vec.Word2Vec(vector_size=10, min_count=0, seed=42, hs=0, negative=5)
    model_hs.build_vocab_from_freq(freq_dict)
    model_neg.build_vocab_from_freq(freq_dict)
    self.assertEqual(len(model_hs.wv), 12)
    self.assertEqual(len(model_neg.wv), 12)
    for k in freq_dict_orig.keys():
        self.assertEqual(model_hs.wv.get_vecattr(k, 'count'), freq_dict_orig[k])
        self.assertEqual(model_neg.wv.get_vecattr(k, 'count'), freq_dict_orig[k])
    new_freq_dict = {'computer': 1, 'artificial': 4, 'human': 1, 'graph': 1, 'intelligence': 4, 'system': 1, 'trees': 1}
    model_hs.build_vocab_from_freq(new_freq_dict, update=True)
    model_neg.build_vocab_from_freq(new_freq_dict, update=True)
    self.assertEqual(model_hs.wv.get_vecattr('graph', 'count'), 4)
    self.assertEqual(model_hs.wv.get_vecattr('artificial', 'count'), 4)
    self.assertEqual(len(model_hs.wv), 14)
    self.assertEqual(len(model_neg.wv), 14)