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
def test_minimize_l_bfgs_b(self):
    opts = {'disp': False, 'maxiter': self.maxiter}
    r = optimize.minimize(self.func, self.startparams, method='L-BFGS-B', jac=self.grad, options=opts)
    assert_allclose(self.func(r.x), self.func(self.solution), atol=1e-06)
    assert self.gradcalls == r.njev
    self.funccalls = self.gradcalls = 0
    ra = optimize.minimize(self.func, self.startparams, method='L-BFGS-B', options=opts)
    assert self.funccalls == ra.nfev
    assert_allclose(self.func(ra.x), self.func(self.solution), atol=1e-06)
    self.funccalls = self.gradcalls = 0
    ra = optimize.minimize(self.func, self.startparams, jac='3-point', method='L-BFGS-B', options=opts)
    assert self.funccalls == ra.nfev
    assert_allclose(self.func(ra.x), self.func(self.solution), atol=1e-06)