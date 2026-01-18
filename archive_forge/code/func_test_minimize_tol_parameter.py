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
def test_minimize_tol_parameter(self):

    def func(z):
        x, y = z
        return x ** 2 * y ** 2 + x ** 4 + 1

    def dfunc(z):
        x, y = z
        return np.array([2 * x * y ** 2 + 4 * x ** 3, 2 * x ** 2 * y])
    for method in ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp']:
        if method in ('nelder-mead', 'powell', 'cobyla'):
            jac = None
        else:
            jac = dfunc
        sol1 = optimize.minimize(func, [1, 1], jac=jac, tol=1e-10, method=method)
        sol2 = optimize.minimize(func, [1, 1], jac=jac, tol=1.0, method=method)
        assert func(sol1.x) < func(sol2.x), f'{method}: {func(sol1.x)} vs. {func(sol2.x)}'