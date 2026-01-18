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
def test_line_for_search():
    line_for_search = optimize._optimize._line_for_search
    lower_bound = np.array([-5.3, -1, -1.5, -3])
    upper_bound = np.array([1.9, 1, 2.8, 3])
    x0 = np.array([0.0, 0, 0, 0])
    x1 = np.array([0.0, 2, -3, 0])
    all_tests = ((x0, np.array([1.0, 0, 0, 0]), -5.3, 1.9), (x0, np.array([0.0, 1, 0, 0]), -1, 1), (x0, np.array([0.0, 0, 1, 0]), -1.5, 2.8), (x0, np.array([0.0, 0, 0, 1]), -3, 3), (x0, np.array([1.0, 1, 0, 0]), -1, 1), (x0, np.array([1.0, 0, -1, 2]), -1.5, 1.5), (x0, np.array([2.0, 0, -1, 2]), -1.5, 0.95), (x1, np.array([1.0, 0, 0, 0]), -5.3, 1.9), (x1, np.array([0.0, 1, 0, 0]), -3, -1), (x1, np.array([0.0, 0, 1, 0]), 1.5, 5.8), (x1, np.array([0.0, 0, 0, 1]), -3, 3), (x1, np.array([1.0, 1, 0, 0]), -3, -1), (x1, np.array([1.0, 0, -1, 0]), -5.3, -1.5))
    for x, alpha, lmin, lmax in all_tests:
        mi, ma = line_for_search(x, alpha, lower_bound, upper_bound)
        assert_allclose(mi, lmin, atol=1e-06)
        assert_allclose(ma, lmax, atol=1e-06)
    lower_bound = np.array([-np.inf, -1, -np.inf, -3])
    upper_bound = np.array([np.inf, 1, 2.8, np.inf])
    all_tests = ((x0, np.array([1.0, 0, 0, 0]), -np.inf, np.inf), (x0, np.array([0.0, 1, 0, 0]), -1, 1), (x0, np.array([0.0, 0, 1, 0]), -np.inf, 2.8), (x0, np.array([0.0, 0, 0, 1]), -3, np.inf), (x0, np.array([1.0, 1, 0, 0]), -1, 1), (x0, np.array([1.0, 0, -1, 2]), -1.5, np.inf), (x1, np.array([1.0, 0, 0, 0]), -np.inf, np.inf), (x1, np.array([0.0, 1, 0, 0]), -3, -1), (x1, np.array([0.0, 0, 1, 0]), -np.inf, 5.8), (x1, np.array([0.0, 0, 0, 1]), -3, np.inf), (x1, np.array([1.0, 1, 0, 0]), -3, -1), (x1, np.array([1.0, 0, -1, 0]), -5.8, np.inf))
    for x, alpha, lmin, lmax in all_tests:
        mi, ma = line_for_search(x, alpha, lower_bound, upper_bound)
        assert_allclose(mi, lmin, atol=1e-06)
        assert_allclose(ma, lmax, atol=1e-06)