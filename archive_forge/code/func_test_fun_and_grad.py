import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_fun_and_grad(self):
    ex = ExScalarFunction()

    def fg_allclose(x, y):
        assert_allclose(x[0], y[0])
        assert_allclose(x[1], y[1])
    x0 = [2.0, 0.3]
    analit = ScalarFunction(ex.fun, x0, (), ex.grad, ex.hess, None, (-np.inf, np.inf))
    fg = (ex.fun(x0), ex.grad(x0))
    fg_allclose(analit.fun_and_grad(x0), fg)
    assert analit.ngev == 1
    x0[1] = 1.0
    fg = (ex.fun(x0), ex.grad(x0))
    fg_allclose(analit.fun_and_grad(x0), fg)
    x0 = [2.0, 0.3]
    sf = ScalarFunction(ex.fun, x0, (), '3-point', ex.hess, None, (-np.inf, np.inf))
    assert sf.ngev == 1
    fg = (ex.fun(x0), ex.grad(x0))
    fg_allclose(sf.fun_and_grad(x0), fg)
    assert sf.ngev == 1
    x0[1] = 1.0
    fg = (ex.fun(x0), ex.grad(x0))
    fg_allclose(sf.fun_and_grad(x0), fg)