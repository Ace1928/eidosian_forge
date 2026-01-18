import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_LinearVectorFunction():
    A_dense = np.array([[-1, 2, 0], [0, 4, 2]])
    x0 = np.zeros(3)
    A_sparse = csr_matrix(A_dense)
    x = np.array([1, -1, 0])
    v = np.array([-1, 1])
    Ax = np.array([-3, -4])
    f1 = LinearVectorFunction(A_dense, x0, None)
    assert_(not f1.sparse_jacobian)
    f2 = LinearVectorFunction(A_dense, x0, True)
    assert_(f2.sparse_jacobian)
    f3 = LinearVectorFunction(A_dense, x0, False)
    assert_(not f3.sparse_jacobian)
    f4 = LinearVectorFunction(A_sparse, x0, None)
    assert_(f4.sparse_jacobian)
    f5 = LinearVectorFunction(A_sparse, x0, True)
    assert_(f5.sparse_jacobian)
    f6 = LinearVectorFunction(A_sparse, x0, False)
    assert_(not f6.sparse_jacobian)
    assert_array_equal(f1.fun(x), Ax)
    assert_array_equal(f2.fun(x), Ax)
    assert_array_equal(f1.jac(x), A_dense)
    assert_array_equal(f2.jac(x).toarray(), A_sparse.toarray())
    assert_array_equal(f1.hess(x, v).toarray(), np.zeros((3, 3)))