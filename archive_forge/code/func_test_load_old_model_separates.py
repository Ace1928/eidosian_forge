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
def test_load_old_model_separates(self):
    """Test loading an old word2vec model of indeterminate version"""
    model_file = 'word2vec_old_sep'
    model = word2vec.Word2Vec.load(datapath(model_file))
    self.assertTrue(model.wv.vectors.shape == (12, 100))
    self.assertTrue(len(model.wv) == 12)
    self.assertTrue(len(model.wv.index_to_key) == 12)
    self.assertTrue(model.syn1neg.shape == (len(model.wv), model.wv.vector_size))
    self.assertTrue(len(model.wv.vectors_lockf.shape) > 0)
    self.assertTrue(model.cum_table.shape == (12,))
    self.onlineSanity(model, trained_model=True)