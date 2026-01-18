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
def test_bounded_powell_outsidebounds():

    def func(x):
        return np.sum(x ** 2)
    bounds = ((-1, 1), (-1, 1), (-1, 1))
    x0 = [-4, 0.5, -0.8]
    with assert_warns(optimize.OptimizeWarning):
        res = optimize.minimize(func, x0, bounds=bounds, method='Powell')
    assert_allclose(res.x, np.array([0.0] * len(x0)), atol=1e-06)
    assert_equal(res.success, True)
    assert_equal(res.status, 0)
    direc = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    with assert_warns(optimize.OptimizeWarning):
        res = optimize.minimize(func, x0, bounds=bounds, method='Powell', options={'direc': direc})
    assert_allclose(res.x, np.array([-4.0, 0, 0]), atol=1e-06)
    assert_equal(res.success, False)
    assert_equal(res.status, 4)