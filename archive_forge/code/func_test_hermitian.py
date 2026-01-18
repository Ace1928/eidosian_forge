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
def test_hermitian():
    """Check complex-value Hermitian cases.
    """
    rnd = np.random.RandomState(0)
    sizes = [3, 12]
    ks = [1, 2]
    gens = [True, False]
    for s, k, gen, dh, dx, db in itertools.product(sizes, ks, gens, gens, gens, gens):
        H = rnd.random((s, s)) + 1j * rnd.random((s, s))
        H = 10 * np.eye(s) + H + H.T.conj()
        H = H.astype(np.complex128) if dh else H.astype(np.complex64)
        X = rnd.standard_normal((s, k))
        X = X + 1j * rnd.standard_normal((s, k))
        X = X.astype(np.complex128) if dx else X.astype(np.complex64)
        if not gen:
            B = np.eye(s)
            w, v = lobpcg(H, X, maxiter=99, verbosityLevel=0)
            wb, _ = lobpcg(H, X, B, maxiter=99, verbosityLevel=0)
            assert_allclose(w, wb, rtol=1e-06)
            w0, _ = eigh(H)
        else:
            B = rnd.random((s, s)) + 1j * rnd.random((s, s))
            B = 10 * np.eye(s) + B.dot(B.T.conj())
            B = B.astype(np.complex128) if db else B.astype(np.complex64)
            w, v = lobpcg(H, X, B, maxiter=99, verbosityLevel=0)
            w0, _ = eigh(H, B)
        for wx, vx in zip(w, v.T):
            assert_allclose(np.linalg.norm(H.dot(vx) - B.dot(vx) * wx) / np.linalg.norm(H.dot(vx)), 0, atol=0.05, rtol=0)
            j = np.argmin(abs(w0 - wx))
            assert_allclose(wx, w0[j], rtol=0.0001)