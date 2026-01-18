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
def test_bfgs_nan_return(self):

    def func(x):
        return np.nan
    with np.errstate(invalid='ignore'):
        result = optimize.minimize(func, 0)
    assert np.isnan(result['fun'])
    assert result['success'] is False

    def func(x):
        return 0 if x == 0 else np.nan

    def fprime(x):
        return np.ones_like(x)
    with np.errstate(invalid='ignore'):
        result = optimize.minimize(func, 0, jac=fprime)
    assert np.isnan(result['fun'])
    assert result['success'] is False