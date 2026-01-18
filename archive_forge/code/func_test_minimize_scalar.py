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
def test_minimize_scalar(self):
    x = optimize.minimize_scalar(self.fun).x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, method='Brent')
    assert x.success
    x = optimize.minimize_scalar(self.fun, method='Brent', options=dict(maxiter=3))
    assert not x.success
    x = optimize.minimize_scalar(self.fun, bracket=(-3, -2), args=(1.5,), method='Brent').x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, method='Brent', args=(1.5,)).x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15), args=(1.5,), method='Brent').x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, bracket=(-3, -2), args=(1.5,), method='golden').x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, method='golden', args=(1.5,)).x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, bracket=(-15, -1, 15), args=(1.5,), method='golden').x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, bounds=(0, 1), args=(1.5,), method='Bounded').x
    assert_allclose(x, 1, atol=0.0001)
    x = optimize.minimize_scalar(self.fun, bounds=(1, 5), args=(1.5,), method='bounded').x
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.minimize_scalar(self.fun, bounds=(np.array([1]), np.array([5])), args=(np.array([1.5]),), method='bounded').x
    assert_allclose(x, self.solution, atol=1e-06)
    assert_raises(ValueError, optimize.minimize_scalar, self.fun, bounds=(5, 1), method='bounded', args=(1.5,))
    assert_raises(ValueError, optimize.minimize_scalar, self.fun, bounds=(np.zeros(2), 1), method='bounded', args=(1.5,))
    x = optimize.minimize_scalar(self.fun, bounds=(1, np.array(5)), method='bounded').x
    assert_allclose(x, self.solution, atol=1e-06)