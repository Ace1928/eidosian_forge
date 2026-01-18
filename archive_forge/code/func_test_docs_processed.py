import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
def test_docs_processed(self):
    self.assertEqual(self.model.docs_processed, 9)
    self.assertEqual(self.model.docs_processed, self.corpus.num_docs)