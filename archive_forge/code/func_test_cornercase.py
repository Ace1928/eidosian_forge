from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from platform import python_implementation
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import lgmres, gmres
def test_cornercase(self):
    np.random.seed(1234)
    for n in [3, 5, 10, 100]:
        A = 2 * eye(n)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, '.*called without specifying.*')
            b = np.ones(n)
            x, info = lgmres(A, b, maxiter=10)
            assert_equal(info, 0)
            assert_allclose(A.dot(x) - b, 0, atol=1e-14)
            x, info = lgmres(A, b, rtol=0, maxiter=10)
            if info == 0:
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)
            b = np.random.rand(n)
            x, info = lgmres(A, b, maxiter=10)
            assert_equal(info, 0)
            assert_allclose(A.dot(x) - b, 0, atol=1e-14)
            x, info = lgmres(A, b, rtol=0, maxiter=10)
            if info == 0:
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)