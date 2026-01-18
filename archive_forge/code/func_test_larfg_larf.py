import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_larfg_larf():
    np.random.seed(1234)
    a0 = np.random.random((4, 4))
    a0 = a0.T.dot(a0)
    a0j = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
    a0j = a0j.T.conj().dot(a0j)
    for dtype in 'fdFD':
        larfg, larf = get_lapack_funcs(['larfg', 'larf'], dtype=dtype)
        if dtype in 'FD':
            a = a0j.copy()
        else:
            a = a0.copy()
        alpha, x, tau = larfg(a.shape[0] - 1, a[1, 0], a[2:, 0])
        expected = np.zeros_like(a[:, 0])
        expected[0] = a[0, 0]
        expected[1] = alpha
        v = np.zeros_like(a[1:, 0])
        v[0] = 1.0
        v[1:] = x
        a[1:, :] = larf(v, tau.conjugate(), a[1:, :], np.zeros(a.shape[1]))
        a[:, 1:] = larf(v, tau, a[:, 1:], np.zeros(a.shape[0]), side='R')
        assert_allclose(a[:, 0], expected, atol=1e-05)
        assert_allclose(a[0, :], expected, atol=1e-05)