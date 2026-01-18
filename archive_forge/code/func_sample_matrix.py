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
def sample_matrix(self, m, dtype, order):
    v = self.sample_vector(m, dtype)
    Q = np.diag(v)
    idx = np.nonzero(np.eye(Q.shape[0], Q.shape[1], 1))
    Q[idx] = v[1:]
    idx = np.nonzero(np.eye(Q.shape[0], Q.shape[1], -1))
    Q[idx] = v[:-1]
    Q = np.array(Q, dtype=dtype, order=order)
    return Q