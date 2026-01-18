import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def check_squareform_matrix(self, dtype):
    A = np.zeros((0, 0), dtype=dtype)
    rA = squareform(A)
    assert_equal(rA.shape, (0,))
    assert_equal(rA.dtype, dtype)
    A = np.zeros((1, 1), dtype=dtype)
    rA = squareform(A)
    assert_equal(rA.shape, (0,))
    assert_equal(rA.dtype, dtype)
    A = np.array([[0, 4.2], [4.2, 0]], dtype=dtype)
    rA = squareform(A)
    assert_equal(rA.shape, (1,))
    assert_equal(rA.dtype, dtype)
    assert_array_equal(rA, np.array([4.2], dtype=dtype))