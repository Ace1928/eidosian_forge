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
def test_neldermead_initial_simplex(self):
    simplex = np.zeros((4, 3))
    simplex[...] = self.startparams
    for j in range(3):
        simplex[j + 1, j] += 0.1
    if self.use_wrapper:
        opts = {'maxiter': self.maxiter, 'disp': False, 'return_all': True, 'initial_simplex': simplex}
        res = optimize.minimize(self.func, self.startparams, args=(), method='Nelder-mead', options=opts)
        params, fopt, numiter, func_calls, warnflag = (res['x'], res['fun'], res['nit'], res['nfev'], res['status'])
        assert_allclose(res['allvecs'][0], simplex[0])
    else:
        retval = optimize.fmin(self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=False, retall=False, initial_simplex=simplex)
        params, fopt, numiter, func_calls, warnflag = retval
    assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
    assert self.funccalls == 100, self.funccalls
    assert self.gradcalls == 0, self.gradcalls
    assert_allclose(self.trace[50:52], [[0.14687474, -0.5103282, 0.48252111], [0.14474003, -0.5282084, 0.48743951]], atol=1e-14, rtol=1e-07)