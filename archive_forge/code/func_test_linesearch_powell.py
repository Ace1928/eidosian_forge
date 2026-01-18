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
def test_linesearch_powell():
    linesearch_powell = optimize._optimize._linesearch_powell

    def func(x):
        return np.sum((x - np.array([-1.0, 2.0, 1.5, -0.4])) ** 2)
    p0 = np.array([0.0, 0, 0, 0])
    fval = func(p0)
    lower_bound = np.array([-np.inf] * 4)
    upper_bound = np.array([np.inf] * 4)
    all_tests = ((np.array([1.0, 0, 0, 0]), -1), (np.array([0.0, 1, 0, 0]), 2), (np.array([0.0, 0, 1, 0]), 1.5), (np.array([0.0, 0, 0, 1]), -0.4), (np.array([-1.0, 0, 1, 0]), 1.25), (np.array([0.0, 0, 1, 1]), 0.55), (np.array([2.0, 0, -1, 1]), -0.65))
    for xi, l in all_tests:
        f, p, direction = linesearch_powell(func, p0, xi, fval=fval, tol=1e-05)
        assert_allclose(f, func(l * xi), atol=1e-06)
        assert_allclose(p, l * xi, atol=1e-06)
        assert_allclose(direction, l * xi, atol=1e-06)
        f, p, direction = linesearch_powell(func, p0, xi, tol=1e-05, lower_bound=lower_bound, upper_bound=upper_bound, fval=fval)
        assert_allclose(f, func(l * xi), atol=1e-06)
        assert_allclose(p, l * xi, atol=1e-06)
        assert_allclose(direction, l * xi, atol=1e-06)