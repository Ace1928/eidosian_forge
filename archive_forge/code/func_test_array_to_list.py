import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_array_to_list(self):
    a = self.a
    assert_equal(a.tolist(), [[1, 2], [3, 4]])