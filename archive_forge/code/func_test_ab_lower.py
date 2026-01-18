import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_ab_lower(self):
    f = getattr(fblas, 'dtrmm', None)
    if f is not None:
        result = f(1.0, self.a, self.b, lower=True)
        expected = np.array([[3.0, 4.0, -1.0], [-1.0, -2.0, 0.0]])
        assert_array_almost_equal(result, expected)