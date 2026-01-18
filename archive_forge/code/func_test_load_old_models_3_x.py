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
def test_load_old_models_3_x(self):
    """Test loading 3.x models"""
    model_file = 'word2vec_3.3'
    model = word2vec.Word2Vec.load(datapath(model_file))
    self.assertEqual(model.max_final_vocab, None)
    self.assertEqual(model.max_final_vocab, None)
    old_versions = ['3.0.0', '3.1.0', '3.2.0', '3.3.0', '3.4.0']
    for old_version in old_versions:
        self._check_old_version(old_version)