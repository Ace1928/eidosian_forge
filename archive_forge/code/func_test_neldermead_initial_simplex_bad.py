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
def test_neldermead_initial_simplex_bad(self):
    bad_simplices = []
    simplex = np.zeros((3, 2))
    simplex[...] = self.startparams[:2]
    for j in range(2):
        simplex[j + 1, j] += 0.1
    bad_simplices.append(simplex)
    simplex = np.zeros((3, 3))
    bad_simplices.append(simplex)
    for simplex in bad_simplices:
        if self.use_wrapper:
            opts = {'maxiter': self.maxiter, 'disp': False, 'return_all': False, 'initial_simplex': simplex}
            assert_raises(ValueError, optimize.minimize, self.func, self.startparams, args=(), method='Nelder-mead', options=opts)
        else:
            assert_raises(ValueError, optimize.fmin, self.func, self.startparams, args=(), maxiter=self.maxiter, full_output=True, disp=False, retall=False, initial_simplex=simplex)