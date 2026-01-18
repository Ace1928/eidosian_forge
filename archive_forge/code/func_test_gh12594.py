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
def test_gh12594():

    def f(x):
        return x[0] ** 2 + (x[1] - 1) ** 2
    bounds = Bounds(lb=[-10, -10], ub=[10, 10])
    res = optimize.minimize(f, x0=(0, 0), method='Powell', bounds=bounds)
    bounds = Bounds(lb=np.array([-10, -10]), ub=np.array([10, 10]))
    ref = optimize.minimize(f, x0=(0, 0), method='Powell', bounds=bounds)
    assert_allclose(res.fun, ref.fun)
    assert_allclose(res.x, ref.x)