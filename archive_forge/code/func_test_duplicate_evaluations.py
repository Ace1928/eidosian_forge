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
@pytest.mark.parametrize('method', ['nelder-mead', 'cg', 'bfgs', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'])
def test_duplicate_evaluations(self, method):
    jac = hess = None
    if method in ('newton-cg', 'trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg'):
        jac = self.grad
    if method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg'):
        hess = self.hess
    with np.errstate(invalid='ignore'), suppress_warnings() as sup:
        sup.filter(UserWarning, 'delta_grad == 0.*')
        optimize.minimize(self.func, self.startparams, method=method, jac=jac, hess=hess)
    for i in range(1, len(self.trace)):
        if np.array_equal(self.trace[i - 1], self.trace[i]):
            raise RuntimeError(f'Duplicate evaluations made by {method}')