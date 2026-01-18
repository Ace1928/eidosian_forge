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
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('matrix_size', [(3, 4), (7, 6), (6, 6)])
def test_geqrfp(dtype, matrix_size):
    np.random.seed(42)
    rtol = 250 * np.finfo(dtype).eps
    atol = 100 * np.finfo(dtype).eps
    geqrfp = get_lapack_funcs('geqrfp', dtype=dtype)
    gqr = get_lapack_funcs('orgqr', dtype=dtype)
    m, n = matrix_size
    A = generate_random_dtype_array((m, n), dtype=dtype)
    qr_A, tau, info = geqrfp(A)
    r = np.triu(qr_A)
    if m > n:
        qqr = np.zeros((m, m), dtype=dtype)
        qqr[:, :n] = qr_A
        q = gqr(qqr, tau=tau, lwork=m)[0]
    else:
        q = gqr(qr_A[:, :m], tau=tau, lwork=m)[0]
    assert_allclose(q @ r, A, rtol=rtol)
    assert_allclose(np.eye(q.shape[0]), q @ q.conj().T, rtol=rtol, atol=atol)
    assert_allclose(r, np.triu(r), rtol=rtol)
    assert_(np.all(np.diag(r) > np.zeros(len(np.diag(r)))))
    assert_(info == 0)
    A_negative = generate_random_dtype_array((n, m), dtype=dtype) * -1
    r_rq_neg, q_rq_neg = qr(A_negative)
    rq_A_neg, tau_neg, info_neg = geqrfp(A_negative)
    assert_(np.any(np.diag(r_rq_neg) < 0) and np.all(np.diag(r) > 0))