import logging
import unittest
from gensim import matutils
from scipy.sparse import csr_matrix
import numpy as np
import math
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import ldamodel
from gensim.test.utils import datapath, common_dictionary, common_corpus
def test_bow(self):
    potentialbow = [(0, 0.4)]
    result = matutils.isbow(potentialbow)
    expected = True
    self.assertEqual(expected, result)
    potentialbow = [(0, 4.0), (1, 2.0), (2, 5.0), (3, 8.0)]
    result = matutils.isbow(potentialbow)
    expected = True
    self.assertEqual(expected, result)
    potentialbow = []
    result = matutils.isbow(potentialbow)
    expected = True
    self.assertEqual(expected, result)
    potentialbow = [[(2, 1), (3, 1), (4, 1), (5, 1), (1, 1), (7, 1)]]
    result = matutils.isbow(potentialbow)
    expected = False
    self.assertEqual(expected, result)
    potentialbow = [(1, 3, 6)]
    result = matutils.isbow(potentialbow)
    expected = False
    self.assertEqual(expected, result)
    potentialbow = csr_matrix([[1, 0.4], [0, 0.3], [2, 0.1]])
    result = matutils.isbow(potentialbow)
    expected = True
    self.assertEqual(expected, result)
    potentialbow = np.array([[1, 0.4], [0, 0.2], [2, 0.2]])
    result = matutils.isbow(potentialbow)
    expected = True
    self.assertEqual(expected, result)