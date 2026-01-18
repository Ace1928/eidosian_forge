import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
def test_bounded_powell_vs_powell():

    def func(x):
        return np.sum(x ** 2)
    bounds = ((-5, -1), (-10, -0.1), (1, 9.2), (-4, 7.6), (-15.9, -2))
    x0 = [-2.1, -5.2, 1.9, 0, -2]
    options = {'ftol': 1e-10, 'xtol': 1e-10}
    res_powell = optimize.minimize(func, x0, method='Powell', options=options)
    assert_allclose(res_powell.x, 0.0, atol=1e-06)
    assert_allclose(res_powell.fun, 0.0, atol=1e-06)
    res_bounded_powell = optimize.minimize(func, x0, options=options, bounds=bounds, method='Powell')
    p = np.array([-1, -0.1, 1, 0, -2])
    assert_allclose(res_bounded_powell.x, p, atol=1e-06)
    assert_allclose(res_bounded_powell.fun, func(p), atol=1e-06)
    bounds = ((None, -1), (-np.inf, -0.1), (1, np.inf), (-4, None), (-15.9, -2))
    res_bounded_powell = optimize.minimize(func, x0, options=options, bounds=bounds, method='Powell')
    p = np.array([-1, -0.1, 1, 0, -2])
    assert_allclose(res_bounded_powell.x, p, atol=1e-06)
    assert_allclose(res_bounded_powell.fun, func(p), atol=1e-06)

    def func(x):
        t = np.sin(-x[0]) * np.cos(x[1]) * np.sin(-x[0] * x[1]) * np.cos(x[1])
        t -= np.cos(np.sin(x[1] * x[2]) * np.cos(x[2]))
        return t ** 2
    bounds = [(-2, 5)] * 3
    x0 = [-0.5, -0.5, -0.5]
    res_powell = optimize.minimize(func, x0, method='Powell')
    res_bounded_powell = optimize.minimize(func, x0, bounds=bounds, method='Powell')
    assert_allclose(res_powell.fun, 0.007136253919761627, atol=1e-06)
    assert_allclose(res_bounded_powell.fun, 0, atol=1e-06)
    bounds = [(-np.inf, np.inf)] * 3
    res_bounded_powell = optimize.minimize(func, x0, bounds=bounds, method='Powell')
    assert_allclose(res_powell.fun, res_bounded_powell.fun, atol=1e-06)
    assert_allclose(res_powell.nfev, res_bounded_powell.nfev, atol=1e-06)
    assert_allclose(res_powell.x, res_bounded_powell.x, atol=1e-06)
    x0 = [45.46254415, -26.52351498, 31.74830248]
    bounds = [(-2, 5)] * 3
    with assert_warns(optimize.OptimizeWarning):
        res_bounded_powell = optimize.minimize(func, x0, bounds=bounds, method='Powell')
    assert_allclose(res_bounded_powell.fun, 0, atol=1e-06)