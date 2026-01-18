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
def test_minimize_l_bfgs_b_ftol(self):
    v0 = None
    for tol in [0.1, 0.0001, 1e-07, 1e-10]:
        opts = {'disp': False, 'maxiter': self.maxiter, 'ftol': tol}
        sol = optimize.minimize(self.func, self.startparams, method='L-BFGS-B', jac=self.grad, options=opts)
        v = self.func(sol.x)
        if v0 is None:
            v0 = v
        else:
            assert v < v0
        assert_allclose(v, self.func(self.solution), rtol=tol)