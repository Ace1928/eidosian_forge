import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_member_ravel(self):
    assert_equal(self.a.ravel().shape, (2,))
    assert_equal(self.m.ravel().shape, (1, 2))