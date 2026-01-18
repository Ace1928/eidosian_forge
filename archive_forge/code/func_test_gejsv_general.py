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
@pytest.mark.parametrize('size', [(6, 5), (5, 5)])
@pytest.mark.parametrize('dtype', REAL_DTYPES)
@pytest.mark.parametrize('joba', range(6))
@pytest.mark.parametrize('jobu', range(4))
@pytest.mark.parametrize('jobv', range(4))
@pytest.mark.parametrize('jobr', [0, 1])
@pytest.mark.parametrize('jobp', [0, 1])
def test_gejsv_general(size, dtype, joba, jobu, jobv, jobr, jobp, jobt=0):
    """Test the lapack routine ?gejsv.

    This function tests that a singular value decomposition can be performed
    on the random M-by-N matrix A. The test performs the SVD using ?gejsv
    then performs the following checks:

    * ?gejsv exist successfully (info == 0)
    * The returned singular values are correct
    * `A` can be reconstructed from `u`, `SIGMA`, `v`
    * Ensure that u.T @ u is the identity matrix
    * Ensure that v.T @ v is the identity matrix
    * The reported matrix rank
    * The reported number of singular values
    * If denormalized floats are required

    Notes
    -----
    joba specifies several choices effecting the calculation's accuracy
    Although all arguments are tested, the tests only check that the correct
    solution is returned - NOT that the prescribed actions are performed
    internally.

    jobt is, as of v3.9.0, still experimental and removed to cut down number of
    test cases. However keyword itself is tested externally.
    """
    seed(42)
    m, n = size
    atol = 100 * np.finfo(dtype).eps
    A = generate_random_dtype_array(size, dtype)
    gejsv = get_lapack_funcs('gejsv', dtype=dtype)
    lsvec = jobu < 2
    rsvec = jobv < 2
    l2tran = jobt == 1 and m == n
    is_complex = np.iscomplexobj(A)
    invalid_real_jobv = jobv == 1 and (not lsvec) and (not is_complex)
    invalid_cplx_jobu = jobu == 2 and (not (rsvec and l2tran)) and is_complex
    invalid_cplx_jobv = jobv == 2 and (not (lsvec and l2tran)) and is_complex
    if invalid_cplx_jobu:
        exit_status = -2
    elif invalid_real_jobv or invalid_cplx_jobv:
        exit_status = -3
    else:
        exit_status = 0
    if jobu > 1 and jobv == 1:
        assert_raises(Exception, gejsv, A, joba, jobu, jobv, jobr, jobt, jobp)
    else:
        sva, u, v, work, iwork, info = gejsv(A, joba=joba, jobu=jobu, jobv=jobv, jobr=jobr, jobt=jobt, jobp=jobp)
        assert_equal(info, exit_status)
        if not exit_status:
            sigma = work[0] / work[1] * sva[:n]
            assert_allclose(sigma, svd(A, compute_uv=False), atol=atol)
            if jobu == 1:
                u = u[:, :n]
            if lsvec and rsvec:
                assert_allclose(u @ np.diag(sigma) @ v.conj().T, A, atol=atol)
            if lsvec:
                assert_allclose(u.conj().T @ u, np.identity(n), atol=atol)
            if rsvec:
                assert_allclose(v.conj().T @ v, np.identity(n), atol=atol)
            assert_equal(iwork[0], np.linalg.matrix_rank(A))
            assert_equal(iwork[1], np.count_nonzero(sigma))
            assert_equal(iwork[2], 0)