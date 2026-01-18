import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_finite_difference_hess_linear_operator(self):
    ex = ExVectorialFunction()
    nfev = 0
    njev = 0
    nhev = 0
    x0 = [1.0, 0.0]
    v0 = [1.0, 2.0]
    analit = VectorFunction(ex.fun, x0, ex.jac, ex.hess, None, None, (-np.inf, np.inf), None)
    nfev += 1
    njev += 1
    nhev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev, nhev)
    approx = VectorFunction(ex.fun, x0, ex.jac, '2-point', None, None, (-np.inf, np.inf), None)
    assert_(isinstance(approx.H, LinearOperator))
    for p in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
        assert_array_equal(analit.f, approx.f)
        assert_array_almost_equal(analit.J, approx.J)
        assert_array_almost_equal(analit.H.dot(p), approx.H.dot(p))
    nfev += 1
    njev += 4
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    x = [2.0, 1.0]
    H_analit = analit.hess(x, v0)
    nhev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    H_approx = approx.hess(x, v0)
    assert_(isinstance(H_approx, LinearOperator))
    for p in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
        assert_array_almost_equal(H_analit.dot(p), H_approx.dot(p), decimal=5)
    njev += 4
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    x = [2.1, 1.2]
    v = [1.0, 1.0]
    H_analit = analit.hess(x, v)
    nhev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    H_approx = approx.hess(x, v)
    assert_(isinstance(H_approx, LinearOperator))
    for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
        assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
    njev += 4
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    x = [2.5, 0.3]
    _ = analit.jac(x)
    H_analit = analit.hess(x, v0)
    njev += 1
    nhev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    _ = approx.jac(x)
    H_approx = approx.hess(x, v0)
    assert_(isinstance(H_approx, LinearOperator))
    for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
        assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v), decimal=4)
    njev += 4
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    x = [5.2, 2.3]
    v = [2.3, 5.2]
    _ = analit.jac(x)
    H_analit = analit.hess(x, v)
    njev += 1
    nhev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)
    _ = approx.jac(x)
    H_approx = approx.hess(x, v)
    assert_(isinstance(H_approx, LinearOperator))
    for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
        assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v), decimal=4)
    njev += 4
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(ex.nhev, nhev)
    assert_array_equal(analit.nhev + approx.nhev, nhev)