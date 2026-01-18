import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_gbmv(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 7
        m = 5
        kl = 1
        ku = 2
        A = toeplitz(append(rand(kl + 1), zeros(m - kl - 1)), append(rand(ku + 1), zeros(n - ku - 1)))
        A = A.astype(dtype)
        Ab = zeros((kl + ku + 1, n), dtype=dtype)
        Ab[2, :5] = A[0, 0]
        Ab[1, 1:6] = A[0, 1]
        Ab[0, 2:7] = A[0, 2]
        Ab[3, :4] = A[1, 0]
        x = rand(n).astype(dtype)
        y = rand(m).astype(dtype)
        alpha, beta = (dtype(3), dtype(-5))
        func, = get_blas_funcs(('gbmv',), dtype=dtype)
        y1 = func(m=m, n=n, ku=ku, kl=kl, alpha=alpha, a=Ab, x=x, y=y, beta=beta)
        y2 = alpha * A.dot(x) + beta * y
        assert_array_almost_equal(y1, y2)
        y1 = func(m=m, n=n, ku=ku, kl=kl, alpha=alpha, a=Ab, x=y, y=x, beta=beta, trans=1)
        y2 = alpha * A.T.dot(y) + beta * x
        assert_array_almost_equal(y1, y2)