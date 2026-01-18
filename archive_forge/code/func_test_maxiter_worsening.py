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
@pytest.mark.xfail(reason='see gh-18697')
def test_maxiter_worsening(solver):
    if solver not in (gmres, lgmres, qmr):
        pytest.skip('Solver breakdown case')
    if solver is gmres and platform.machine() == 'aarch64' and (sys.version_info[1] == 9):
        pytest.xfail(reason='gh-13019')
    if solver is lgmres and platform.machine() not in ['x86_64x86', 'aarch64', 'arm64']:
        pytest.xfail(reason='fails on at least ppc64le, ppc64 and riscv64')
    A = np.array([[-0.1112795288033378, 0, 0, 0.16127952880333685], [0, -0.13627952880333782 + 6.283185307179586j, 0, 0], [0, 0, -0.13627952880333782 - 6.283185307179586j, 0], [0.1112795288033368, 0j, 0j, -0.16127952880333785]])
    v = np.ones(4)
    best_error = np.inf
    slack_tol = 9
    for maxiter in range(1, 20):
        x, info = solver(A, v, maxiter=maxiter, rtol=1e-08, atol=0)
        if info == 0:
            assert norm(A @ x - v) <= 1e-08 * norm(v)
        error = np.linalg.norm(A @ x - v)
        best_error = min(best_error, error)
        assert error <= slack_tol * best_error