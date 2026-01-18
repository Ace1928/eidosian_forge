import os
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.linalg._svdp import _svdp
from scipy.sparse import csr_matrix, csc_matrix
def check_svdp(n, m, constructor, dtype, k, irl_mode, which, f=0.8):
    tol = TOLS[dtype]
    M = generate_matrix(np.asarray, n, m, f, dtype)
    Msp = constructor(M)
    u1, sigma1, vt1 = np.linalg.svd(M, full_matrices=False)
    u2, sigma2, vt2, _ = _svdp(Msp, k=k, which=which, irl_mode=irl_mode, tol=tol)
    if which.upper() == 'SM':
        u1 = np.roll(u1, k, 1)
        vt1 = np.roll(vt1, k, 0)
        sigma1 = np.roll(sigma1, k)
    assert_allclose(sigma1[:k], sigma2, rtol=tol, atol=tol)
    assert_orthogonal(u1, u2, rtol=tol, atol=tol)
    assert_orthogonal(vt1.T, vt2.T, rtol=tol, atol=tol)