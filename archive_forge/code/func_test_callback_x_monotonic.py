import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_callback_x_monotonic(self):
    np.random.seed(1)
    A = np.random.rand(20, 20) + np.eye(20)
    b = np.random.rand(20)
    prev_r = [np.inf]
    count = [0]

    def x_cb(x):
        r = np.linalg.norm(A @ x - b)
        assert r <= prev_r[0]
        prev_r[0] = r
        count[0] += 1
    x, info = gmres(A, b, rtol=1e-06, atol=0, callback=x_cb, maxiter=20, restart=10, callback_type='x')
    assert info == 20
    assert count[0] == 20