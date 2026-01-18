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
def test_geqrt_gemqrt(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
        else:
            A = rand(n, n).astype(dtype)
        tol = 100 * np.spacing(dtype(1.0).real)
        geqrt, gemqrt = get_lapack_funcs(('geqrt', 'gemqrt'), dtype=dtype)
        a, t, info = geqrt(n, A)
        assert info == 0
        v = np.tril(a, -1) + np.eye(n, dtype=dtype)
        Q = np.eye(n, dtype=dtype) - v @ t @ v.T.conj()
        R = np.triu(a)
        assert_allclose(Q.T.conj() @ Q, np.eye(n, dtype=dtype), atol=tol, rtol=0.0)
        assert_allclose(Q @ R, A, atol=tol, rtol=0.0)
        if ind > 1:
            C = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
            transpose = 'C'
        else:
            C = rand(n, n).astype(dtype)
            transpose = 'T'
        for side in ('L', 'R'):
            for trans in ('N', transpose):
                c, info = gemqrt(a, t, C, side=side, trans=trans)
                assert info == 0
                if trans == transpose:
                    q = Q.T.conj()
                else:
                    q = Q
                if side == 'L':
                    qC = q @ C
                else:
                    qC = C @ q
                assert_allclose(c, qC, atol=tol, rtol=0.0)
                if (side, trans) == ('L', 'N'):
                    c_default, info = gemqrt(a, t, C)
                    assert info == 0
                    assert_equal(c_default, c)
        assert_raises(Exception, gemqrt, a, t, C, side='A')
        assert_raises(Exception, gemqrt, a, t, C, trans='A')