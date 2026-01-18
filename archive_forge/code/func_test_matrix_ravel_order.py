import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_matrix_ravel_order(self):
    x = matrix([[1, 2, 3], [4, 5, 6]])
    assert_equal(x.ravel(), [[1, 2, 3, 4, 5, 6]])
    assert_equal(x.ravel(order='F'), [[1, 4, 2, 5, 3, 6]])
    assert_equal(x.T.ravel(), [[1, 4, 2, 5, 3, 6]])
    assert_equal(x.T.ravel(order='A'), [[1, 2, 3, 4, 5, 6]])