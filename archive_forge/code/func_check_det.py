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
def check_det(self, cfunc, a, **kwargs):
    expected = det_matrix(a, **kwargs)
    got = cfunc(a, **kwargs)
    resolution = 5 * np.finfo(a.dtype).resolution
    np.testing.assert_allclose(got, expected, rtol=resolution)
    with self.assertNoNRTLeak():
        cfunc(a, **kwargs)