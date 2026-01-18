import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_log_sum_exp(self):
    rs = self.random_state
    for dtype in [np.float16, np.float32, np.float64]:
        for i in range(self.num_runs):
            input = rs.uniform(-1000, 1000, size=(self.num_topics, 1))
            known_good = logsumexp(input)
            test_values = matutils.logsumexp(input)
            msg = 'logsumexp failed for dtype={}'.format(dtype)
            self.assertTrue(np.allclose(known_good, test_values), msg)