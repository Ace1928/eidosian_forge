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
def setup_test_equal_bounds():
    np.random.seed(0)
    x0 = np.random.rand(4)
    lb = np.array([0, 2, -1, -1.0])
    ub = np.array([3, 2, 2, -1.0])
    i_eb = lb == ub

    def check_x(x, check_size=True, check_values=True):
        if check_size:
            assert x.size == 4
        if check_values:
            assert_allclose(x[i_eb], lb[i_eb])

    def func(x):
        check_x(x)
        return optimize.rosen(x)

    def grad(x):
        check_x(x)
        return optimize.rosen_der(x)

    def callback(x, *args):
        check_x(x)

    def constraint1(x):
        check_x(x, check_values=False)
        return x[0:1] - 1

    def jacobian1(x):
        check_x(x, check_values=False)
        dc = np.zeros_like(x)
        dc[0] = 1
        return dc

    def constraint2(x):
        check_x(x, check_values=False)
        return x[2:3] - 0.5

    def jacobian2(x):
        check_x(x, check_values=False)
        dc = np.zeros_like(x)
        dc[2] = 1
        return dc
    c1a = NonlinearConstraint(constraint1, -np.inf, 0)
    c1b = NonlinearConstraint(constraint1, -np.inf, 0, jacobian1)
    c2a = NonlinearConstraint(constraint2, -np.inf, 0)
    c2b = NonlinearConstraint(constraint2, -np.inf, 0, jacobian2)
    methods = ('L-BFGS-B', 'SLSQP', 'TNC')
    kwds = ({'fun': func, 'jac': False}, {'fun': func, 'jac': grad}, {'fun': lambda x: (func(x), grad(x)), 'jac': True})
    bound_types = (lambda lb, ub: list(zip(lb, ub)), Bounds)
    constraints = ((None, None), ([], []), (c1a, c1b), (c2b, c2b), ([c1b], [c1b]), ([c2a], [c2b]), ([c1a, c2a], [c1b, c2b]), ([c1a, c2b], [c1b, c2b]), ([c1b, c2b], [c1b, c2b]))
    callbacks = (None, callback)
    data = {'methods': methods, 'kwds': kwds, 'bound_types': bound_types, 'constraints': constraints, 'callbacks': callbacks, 'lb': lb, 'ub': ub, 'x0': x0, 'i_eb': i_eb}
    return data