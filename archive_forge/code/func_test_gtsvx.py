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
@pytest.mark.parametrize('trans_bool', [False, True])
@pytest.mark.parametrize('fact', ['F', 'N'])
def test_gtsvx(dtype, trans_bool, fact):
    """
    These tests uses ?gtsvx to solve a random Ax=b system for each dtype.
    It tests that the outputs define an LU matrix, that inputs are unmodified,
    transposal options, incompatible shapes, singular matrices, and
    singular factorizations. It parametrizes DTYPES and the 'fact' value along
    with the fact related inputs.
    """
    seed(42)
    atol = 100 * np.finfo(dtype).eps
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    n = 10
    dl = generate_random_dtype_array((n - 1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    du = generate_random_dtype_array((n - 1,), dtype=dtype)
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    trans = ('T' if dtype in REAL_DTYPES else 'C') if trans_bool else 'N'
    b = (A.conj().T if trans_bool else A) @ x
    inputs_cpy = [dl.copy(), d.copy(), du.copy(), b.copy()]
    dlf_, df_, duf_, du2f_, ipiv_, info_ = gttrf(dl, d, du) if fact == 'F' else [None] * 6
    gtsvx_out = gtsvx(dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_, duf=duf_, du2=du2f_, ipiv=ipiv_)
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
    assert_(info == 0, f'?gtsvx info = {info}, should be zero')
    assert_array_equal(dl, inputs_cpy[0])
    assert_array_equal(d, inputs_cpy[1])
    assert_array_equal(du, inputs_cpy[2])
    assert_array_equal(b, inputs_cpy[3])
    assert_allclose(x, x_soln, atol=atol)
    assert_(hasattr(rcond, '__len__') is not True, f'rcond should be scalar but is {rcond}')
    assert_(ferr.shape[0] == b.shape[1], 'ferr.shape is {} but should be {},'.format(ferr.shape[0], b.shape[1]))
    assert_(berr.shape[0] == b.shape[1], 'berr.shape is {} but should be {},'.format(berr.shape[0], b.shape[1]))