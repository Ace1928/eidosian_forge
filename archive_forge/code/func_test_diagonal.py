import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong
@pytest.mark.filterwarnings('ignore:The problem size')
@pytest.mark.parametrize('n, m, m_excluded', [(30, 4, 3), (4, 2, 0)])
def test_diagonal(n, m, m_excluded):
    """Test ``m - m_excluded`` eigenvalues and eigenvectors of
    diagonal matrices of the size ``n`` varying matrix formats:
    dense array, spare matrix, and ``LinearOperator`` for both
    matrixes in the generalized eigenvalue problem ``Av = cBv``
    and for the preconditioner.
    """
    rnd = np.random.RandomState(0)
    vals = np.arange(1, n + 1, dtype=float)
    A_s = diags([vals], [0], (n, n))
    A_a = A_s.toarray()

    def A_f(x):
        return A_s @ x
    A_lo = LinearOperator(matvec=A_f, matmat=A_f, shape=(n, n), dtype=float)
    B_a = eye(n)
    B_s = csr_matrix(B_a)

    def B_f(x):
        return B_a @ x
    B_lo = LinearOperator(matvec=B_f, matmat=B_f, shape=(n, n), dtype=float)
    M_s = diags([1.0 / vals], [0], (n, n))
    M_a = M_s.toarray()

    def M_f(x):
        return M_s @ x
    M_lo = LinearOperator(matvec=M_f, matmat=M_f, shape=(n, n), dtype=float)
    X = rnd.normal(size=(n, m))
    if m_excluded > 0:
        Y = np.eye(n, m_excluded)
    else:
        Y = None
    for A in [A_a, A_s, A_lo]:
        for B in [B_a, B_s, B_lo]:
            for M in [M_a, M_s, M_lo]:
                eigvals, vecs = lobpcg(A, X, B, M=M, Y=Y, maxiter=40, largest=False)
                assert_allclose(eigvals, np.arange(1 + m_excluded, 1 + m_excluded + m))
                _check_eigen(A, eigvals, vecs, rtol=0.001, atol=0.001)