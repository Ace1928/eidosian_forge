from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def obsolete_test_load_old_models_2_x(self):
    """Test loading 2.x models"""
    old_versions = ['2.0.0', '2.1.0', '2.2.0', '2.3.0']
    for old_version in old_versions:
        self._check_old_version(old_version)