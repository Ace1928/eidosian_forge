import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
@needs_lapack
def test_issue3368(self):
    X = np.array([[1.0, 7.54, 6.52], [1.0, 2.7, 4.0], [1.0, 2.5, 3.8], [1.0, 1.15, 5.64], [1.0, 4.22, 3.27], [1.0, 1.41, 5.7]], order='F')
    X_orig = np.copy(X)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    @jit(nopython=True)
    def f2(X, y, test):
        if test:
            X = X[1:2, :]
        return np.linalg.lstsq(X, y)
    f2(X, y, False)
    np.testing.assert_allclose(X, X_orig)