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
def test_neldermead_adaptive():

    def func(x):
        return np.sum(x ** 2)
    p0 = [0.15746215, 0.48087031, 0.44519198, 0.4223638, 0.61505159, 0.32308456, 0.9692297, 0.4471682, 0.77411992, 0.80441652, 0.35994957, 0.75487856, 0.99973421, 0.65063887, 0.09626474]
    res = optimize.minimize(func, p0, method='Nelder-Mead')
    assert_equal(res.success, False)
    res = optimize.minimize(func, p0, method='Nelder-Mead', options={'adaptive': True})
    assert_equal(res.success, True)