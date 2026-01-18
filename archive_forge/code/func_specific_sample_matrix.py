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
def specific_sample_matrix(self, size, dtype, order, rank=None, condition=None):
    """
        Provides a sample matrix with an optionally specified rank or condition
        number.

        size: (rows, columns), the dimensions of the returned matrix.
        dtype: the dtype for the returned matrix.
        order: the memory layout for the returned matrix, 'F' or 'C'.
        rank: the rank of the matrix, an integer value, defaults to full rank.
        condition: the condition number of the matrix (defaults to 1.)

        NOTE: Only one of rank or condition may be set.
        """
    d_cond = 1.0
    if len(size) != 2:
        raise ValueError('size must be a length 2 tuple.')
    if order not in ['F', 'C']:
        raise ValueError("order must be one of 'F' or 'C'.")
    if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
        raise ValueError('dtype must be a numpy floating point type.')
    if rank is not None and condition is not None:
        raise ValueError('Only one of rank or condition can be specified.')
    if condition is None:
        condition = d_cond
    if condition < 1:
        raise ValueError('Condition number must be >=1.')
    np.random.seed(0)
    m, n = size
    if m < 0 or n < 0:
        raise ValueError('Negative dimensions given for matrix shape.')
    minmn = min(m, n)
    if rank is None:
        rv = minmn
    else:
        if rank <= 0:
            raise ValueError('Rank must be greater than zero.')
        if not isinstance(rank, Integral):
            raise ValueError('Rank must an integer.')
        rv = rank
        if rank > minmn:
            raise ValueError('Rank given greater than full rank.')
    if m == 1 or n == 1:
        if condition != d_cond:
            raise ValueError('Condition number was specified for a vector (always 1.).')
        maxmn = max(m, n)
        Q = self.sample_vector(maxmn, dtype).reshape(m, n)
    else:
        tmp = self.sample_vector(m * m, dtype).reshape(m, m)
        U, _ = np.linalg.qr(tmp)
        tmp = self.sample_vector(n * n, dtype)[::-1].reshape(n, n)
        V, _ = np.linalg.qr(tmp)
        sv = np.linspace(d_cond, condition, rv)
        S = np.zeros((m, n))
        idx = np.nonzero(np.eye(m, n))
        S[idx[0][:rv], idx[1][:rv]] = sv
        Q = np.dot(np.dot(U, S), V.T)
        Q = np.array(Q, dtype=dtype, order=order)
    return Q