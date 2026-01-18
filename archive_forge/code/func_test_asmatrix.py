import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_asmatrix(self):
    A = np.arange(100).reshape(10, 10)
    mA = asmatrix(A)
    A[0, 0] = -10
    assert_(A[0, 0] == mA[0, 0])