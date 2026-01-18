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
def test_eb_constraints():

    def f(x):
        return x[0] ** 3 + x[1] ** 2 + x[2] * x[3]

    def cfun(x):
        return x[0] + x[1] + x[2] + x[3] - 40
    constraints = [{'type': 'ineq', 'fun': cfun}]
    bounds = [(0, 20)] * 4
    bounds[1] = (5, 5)
    optimize.minimize(f, x0=[1, 2, 3, 4], method='SLSQP', bounds=bounds, constraints=constraints)
    assert constraints[0]['fun'] == cfun