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
def test_minimize_multiple_constraints():

    def func(x):
        return np.array([25 - 0.2 * x[0] - 0.4 * x[1] - 0.33 * x[2]])

    def func1(x):
        return np.array([x[1]])

    def func2(x):
        return np.array([x[2]])
    cons = ({'type': 'ineq', 'fun': func}, {'type': 'ineq', 'fun': func1}, {'type': 'ineq', 'fun': func2})

    def f(x):
        return -1 * (x[0] + x[1] + x[2])
    res = optimize.minimize(f, [0, 0, 0], method='SLSQP', constraints=cons)
    assert_allclose(res.x, [125, 0, 0], atol=1e-10)