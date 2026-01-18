import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
def test_large_mmap_compressed(self):
    fname = get_tmpfile('gensim_models_lsi.tst.gz')
    model = self.model
    model.save(fname, sep_limit=0)
    return
    self.assertRaises(IOError, lsimodel.LsiModel.load, fname, mmap='r')