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
def test_get_topic_terms(self):
    topic_terms = self.model.get_topic_terms(1)
    for k, v in topic_terms:
        self.assertTrue(isinstance(k, numbers.Integral))
        self.assertTrue(np.issubdtype(v, np.floating))