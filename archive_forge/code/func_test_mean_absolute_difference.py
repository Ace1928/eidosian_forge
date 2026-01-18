import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_mean_absolute_difference(self):
    rs = self.random_state
    for dtype in [np.float16, np.float32, np.float64]:
        for i in range(self.num_runs):
            input1 = rs.uniform(-10000, 10000, size=(self.num_topics,))
            input2 = rs.uniform(-10000, 10000, size=(self.num_topics,))
            known_good = mean_absolute_difference(input1, input2)
            test_values = matutils.mean_absolute_difference(input1, input2)
            msg = 'mean_absolute_difference failed for dtype={}'.format(dtype)
            self.assertTrue(np.allclose(known_good, test_values), msg)