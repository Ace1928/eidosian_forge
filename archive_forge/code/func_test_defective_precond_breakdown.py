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
def test_defective_precond_breakdown(self):
    M = np.eye(3)
    M[2, 2] = 0
    b = np.array([0, 1, 1])
    x = np.array([1, 0, 0])
    A = np.diag([2, 3, 4])
    x, info = gmres(A, b, x0=x, M=M, rtol=1e-15, atol=0)
    assert not np.isnan(x).any()
    if info == 0:
        assert np.linalg.norm(A @ x - b) <= 1e-15 * np.linalg.norm(b)
    assert_allclose(M @ (A @ x), M @ b)