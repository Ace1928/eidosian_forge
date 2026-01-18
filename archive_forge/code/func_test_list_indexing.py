import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_list_indexing(self):
    A = np.arange(6)
    A.shape = (3, 2)
    x = asmatrix(A)
    assert_array_equal(x[:, [1, 0]], x[:, ::-1])
    assert_array_equal(x[[2, 1, 0], :], x[::-1, :])