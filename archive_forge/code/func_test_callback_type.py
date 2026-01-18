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
def test_callback_type(self):
    np.random.seed(1)
    A = np.random.rand(20, 20)
    b = np.random.rand(20)
    cb_count = [0]

    def pr_norm_cb(r):
        cb_count[0] += 1
        assert isinstance(r, float)

    def x_cb(x):
        cb_count[0] += 1
        assert isinstance(x, np.ndarray)
    cb_count = [0]
    x, info = gmres(A, b, rtol=1e-06, atol=0, callback=pr_norm_cb, maxiter=2, restart=50)
    assert info == 2
    assert cb_count[0] == 2
    cb_count = [0]
    x, info = gmres(A, b, rtol=1e-06, atol=0, callback=pr_norm_cb, maxiter=2, restart=50, callback_type='legacy')
    assert info == 2
    assert cb_count[0] == 2
    cb_count = [0]
    x, info = gmres(A, b, rtol=1e-06, atol=0, callback=pr_norm_cb, maxiter=2, restart=50, callback_type='pr_norm')
    assert info == 0
    assert cb_count[0] > 2
    cb_count = [0]
    x, info = gmres(A, b, rtol=1e-06, atol=0, callback=x_cb, maxiter=2, restart=50, callback_type='x')
    assert info == 0
    assert cb_count[0] == 1