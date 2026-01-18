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
@pytest.mark.filterwarnings('ignore::scipy.sparse.SparseEfficiencyWarning')
def test_leftright_precond(self):
    """Check that QMR works with left and right preconditioners"""
    from scipy.sparse.linalg._dsolve import splu
    from scipy.sparse.linalg._interface import LinearOperator
    n = 100
    dat = ones(n)
    A = spdiags([-2 * dat, 4 * dat, -dat], [-1, 0, 1], n, n)
    b = arange(n, dtype='d')
    L = spdiags([-dat / 2, dat], [-1, 0], n, n)
    U = spdiags([4 * dat, -dat], [0, 1], n, n)
    L_solver = splu(L)
    U_solver = splu(U)

    def L_solve(b):
        return L_solver.solve(b)

    def U_solve(b):
        return U_solver.solve(b)

    def LT_solve(b):
        return L_solver.solve(b, 'T')

    def UT_solve(b):
        return U_solver.solve(b, 'T')
    M1 = LinearOperator((n, n), matvec=L_solve, rmatvec=LT_solve)
    M2 = LinearOperator((n, n), matvec=U_solve, rmatvec=UT_solve)
    rtol = 1e-08
    x, info = qmr(A, b, rtol=rtol, maxiter=15, M1=M1, M2=M2)
    assert info == 0
    assert norm(A @ x - b) <= rtol * norm(b)