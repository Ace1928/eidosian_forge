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
def test_l_bfgs_b_maxiter(self):

    class Callback:

        def __init__(self):
            self.nit = 0
            self.fun = None
            self.x = None

        def __call__(self, x):
            self.x = x
            self.fun = optimize.rosen(x)
            self.nit += 1
    c = Callback()
    res = optimize.minimize(optimize.rosen, [0.0, 0.0], method='l-bfgs-b', callback=c, options={'maxiter': 5})
    assert_equal(res.nit, 5)
    assert_almost_equal(res.x, c.x)
    assert_almost_equal(res.fun, c.fun)
    assert_equal(res.status, 1)
    assert res.success is False
    assert_equal(res.message, 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT')