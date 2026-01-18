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
def obsolete_test_load_old_models_pre_1_0(self):
    """Test loading pre-1.0 models"""
    model_file = 'd2v-lee-v0.13.0'
    model = doc2vec.Doc2Vec.load(datapath(model_file))
    self.model_sanity(model)
    old_versions = ['0.12.0', '0.12.1', '0.12.2', '0.12.3', '0.12.4', '0.13.0', '0.13.1', '0.13.2', '0.13.3', '0.13.4']
    for old_version in old_versions:
        self._check_old_version(old_version)