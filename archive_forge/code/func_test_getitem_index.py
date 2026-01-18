import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_getitem_index(self):
    self.assertListEqual(self.s2c[1], [(0, 2), (1, 5), (2, 8)])