import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_member_flatten(self):
    assert_equal(self.a.flatten().shape, (2,))
    assert_equal(self.m.flatten().shape, (1, 2))