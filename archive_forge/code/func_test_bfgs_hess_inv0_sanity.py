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
def test_bfgs_hess_inv0_sanity(self):
    fun = optimize.rosen
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    opts = {'disp': self.disp, 'hess_inv0': 0.01 * np.eye(5)}
    res = optimize.minimize(fun, x0=x0, method='BFGS', args=(), options=opts)
    res_true = optimize.minimize(fun, x0=x0, method='BFGS', args=(), options={'disp': self.disp})
    assert_allclose(res.fun, res_true.fun, atol=1e-06)