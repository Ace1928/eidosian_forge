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
def test_eta_auto(self):
    model1 = self.class_(corpus, id2word=dictionary, eta='symmetric', passes=10)
    modelauto = self.class_(corpus, id2word=dictionary, eta='auto', passes=10)
    self.assertFalse(np.allclose(model1.eta, modelauto.eta))