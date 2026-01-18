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
@pytest.mark.parametrize('n, atol', [(20, 0.001), (5, 1e-08)])
def test_eigs_consistency(n, atol):
    """Check eigs vs. lobpcg consistency.
    """
    vals = np.arange(1, n + 1, dtype=np.float64)
    A = spdiags(vals, 0, n, n)
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((n, 2))
    lvals, lvecs = lobpcg(A, X, largest=True, maxiter=100)
    vals, _ = eigs(A, k=2)
    _check_eigen(A, lvals, lvecs, atol=atol, rtol=0)
    assert_allclose(np.sort(vals), np.sort(lvals), atol=1e-14)