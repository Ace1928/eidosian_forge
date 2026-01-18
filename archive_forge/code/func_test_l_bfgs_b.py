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
def test_l_bfgs_b(self):
    retval = optimize.fmin_l_bfgs_b(self.func, self.startparams, self.grad, args=(), maxiter=self.maxiter)
    params, fopt, d = retval
    assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
    assert self.funccalls == 7, self.funccalls
    assert self.gradcalls == 5, self.gradcalls
    assert_allclose(self.trace[3:5], [[8.117083e-16, -0.5196198, 0.4897617], [0.0, -0.52489628, 0.48753042]], atol=1e-14, rtol=1e-07)