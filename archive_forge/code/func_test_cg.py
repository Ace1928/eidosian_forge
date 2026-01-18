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
def test_cg(self):
    if self.use_wrapper:
        opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
        res = optimize.minimize(self.func, self.startparams, args=(), method='CG', jac=self.grad, options=opts)
        params, fopt, func_calls, grad_calls, warnflag = (res['x'], res['fun'], res['nfev'], res['njev'], res['status'])
    else:
        retval = optimize.fmin_cg(self.func, self.startparams, self.grad, (), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
        params, fopt, func_calls, grad_calls, warnflag = retval
    assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
    assert self.funccalls == 9, self.funccalls
    assert self.gradcalls == 7, self.gradcalls
    assert_allclose(self.trace[2:4], [[0, -0.5, 0.5], [0, -0.505700028, 0.495985862]], atol=1e-14, rtol=1e-07)