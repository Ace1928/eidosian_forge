import logging
import numbers
import os
import unittest
import copy
import numpy as np
from numpy.testing import assert_allclose
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils, utils
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_texts
def test_dtype_backward_compatibility(self):
    lda_3_0_1_fname = datapath('lda_3_0_1_model')
    test_doc = [(0, 1), (1, 1), (2, 1)]
    expected_topics = [(0, 0.8700588697747518), (1, 0.12994113022524822)]
    model = self.class_.load(lda_3_0_1_fname)
    topics = model[test_doc]
    self.assertTrue(np.allclose(expected_topics, topics))