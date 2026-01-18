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
def obsolete_testLoadPreKeyedVectorModel(self):
    """Test loading pre-KeyedVectors word2vec model"""
    if sys.version_info[:2] == (3, 4):
        model_file_suffix = '_py3_4'
    elif sys.version_info < (3,):
        model_file_suffix = '_py2'
    else:
        model_file_suffix = '_py3'
    model_file = 'word2vec_pre_kv%s' % model_file_suffix
    model = word2vec.Word2Vec.load(datapath(model_file))
    self.assertTrue(model.wv.vectors.shape == (len(model.wv), model.vector_size))
    self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
    model_file = 'word2vec_pre_kv_sep%s' % model_file_suffix
    model = word2vec.Word2Vec.load(datapath(model_file))
    self.assertTrue(model.wv.vectors.shape == (len(model.wv), model.vector_size))
    self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))