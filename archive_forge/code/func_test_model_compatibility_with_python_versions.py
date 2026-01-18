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
def test_model_compatibility_with_python_versions(self):
    fname_model_2_7 = datapath('ldamodel_python_2_7')
    model_2_7 = self.class_.load(fname_model_2_7)
    fname_model_3_5 = datapath('ldamodel_python_3_5')
    model_3_5 = self.class_.load(fname_model_3_5)
    self.assertEqual(model_2_7.num_topics, model_3_5.num_topics)
    self.assertTrue(np.allclose(model_2_7.expElogbeta, model_3_5.expElogbeta))
    tstvec = []
    self.assertTrue(np.allclose(model_2_7[tstvec], model_3_5[tstvec]))
    id2word_2_7 = dict(model_2_7.id2word.iteritems())
    id2word_3_5 = dict(model_3_5.id2word.iteritems())
    self.assertEqual(set(id2word_2_7.keys()), set(id2word_3_5.keys()))