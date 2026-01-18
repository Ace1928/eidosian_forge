import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_make_bool_matrix_from_str(self):
    A = matrix('True; True; False')
    B = matrix([[True], [True], [False]])
    assert_array_equal(A, B)