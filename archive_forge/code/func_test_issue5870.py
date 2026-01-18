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
def test_issue5870(self):

    @jit(nopython=True)
    def some_fn(v):
        return np.linalg.pinv(v[0])
    v_data = np.array([[1.0, 3, 2, 7], [-5, 4, 2, 3], [9, -3, 1, 1], [2, -2, 2, 8]], order='F')
    v_orig = np.copy(v_data)
    reshaped_v = v_data.reshape((1, 4, 4))
    expected = some_fn.py_func(reshaped_v)
    np.testing.assert_allclose(v_data, v_orig)
    got = some_fn(reshaped_v)
    np.testing.assert_allclose(v_data, v_orig)
    np.testing.assert_allclose(expected, got)