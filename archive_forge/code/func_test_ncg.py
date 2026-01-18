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
def test_ncg(self):
    if self.use_wrapper:
        opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
        retval = optimize.minimize(self.func, self.startparams, method='Newton-CG', jac=self.grad, args=(), options=opts)['x']
    else:
        retval = optimize.fmin_ncg(self.func, self.startparams, self.grad, args=(), maxiter=self.maxiter, full_output=False, disp=self.disp, retall=False)
    params = retval
    assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
    assert self.funccalls == 7, self.funccalls
    assert self.gradcalls <= 22, self.gradcalls
    assert_allclose(self.trace[3:5], [[-4.35700753e-07, -0.524869435, 0.48752748], [-4.35700753e-07, -0.524869401, 0.487527774]], atol=1e-06, rtol=1e-07)