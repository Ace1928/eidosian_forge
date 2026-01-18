import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile
def test_corpus_transform(self):
    """Test lsi[corpus] transformation."""
    model = self.model
    got = np.vstack([matutils.sparse2full(doc, 2) for doc in model[self.corpus]])
    expected = np.array([[0.65946639, 0.14211544], [2.02454305, -0.42088759], [1.54655361, 0.32358921], [1.81114125, 0.5890525], [0.9336738, -0.27138939], [0.01274618, -0.49016181], [0.04888203, -1.11294699], [0.08063836, -1.56345594], [0.27381003, -1.34694159]])
    self.assertTrue(np.allclose(abs(got), abs(expected)))