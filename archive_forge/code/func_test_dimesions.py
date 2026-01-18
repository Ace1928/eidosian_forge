import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_dimesions(self):
    a = self.a
    x = a[0]
    assert_equal(x.ndim, 2)