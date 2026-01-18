import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_lowest_x(self):
    x0 = np.array([2, 3, 4])
    sf = ScalarFunction(rosen, x0, (), rosen_der, rosen_hess, None, None)
    sf.fun([1, 1, 1])
    sf.fun(x0)
    sf.fun([1.01, 1, 1.0])
    sf.grad([1.01, 1, 1.0])
    assert_equal(sf._lowest_f, 0.0)
    assert_equal(sf._lowest_x, [1.0, 1.0, 1.0])
    sf = ScalarFunction(rosen, x0, (), '2-point', rosen_hess, None, (-np.inf, np.inf))
    sf.fun([1, 1, 1])
    sf.fun(x0)
    sf.fun([1.01, 1, 1.0])
    sf.grad([1.01, 1, 1.0])
    assert_equal(sf._lowest_f, 0.0)
    assert_equal(sf._lowest_x, [1.0, 1.0, 1.0])