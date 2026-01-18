import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
import pytest
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def check_precond_inverse(solver, case):
    tol = 1e-08

    def inverse(b, which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.toarray()
        return np.linalg.solve(A, b)

    def rinverse(b, which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.toarray()
        return np.linalg.solve(A.T, b)
    matvec_count = [0]

    def matvec(b):
        matvec_count[0] += 1
        return case.A @ b

    def rmatvec(b):
        matvec_count[0] += 1
        return case.A.T @ b
    b = case.b
    x0 = 0 * b
    A = LinearOperator(case.A.shape, matvec, rmatvec=rmatvec)
    precond = LinearOperator(case.A.shape, inverse, rmatvec=rinverse)
    matvec_count = [0]
    x, info = solver(A, b, M=precond, x0=x0, tol=tol)
    assert info == 0
    assert_normclose(case.A @ x, b, tol)
    assert matvec_count[0] <= 3