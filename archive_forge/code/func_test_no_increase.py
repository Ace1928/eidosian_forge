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
@pytest.mark.parametrize('method', ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp'])
def test_no_increase(self, method):

    def func(x):
        return (x - 1) ** 2

    def bad_grad(x):
        return 2 * (x - 1) * -1 - 2
    x0 = np.array([2.0])
    f0 = func(x0)
    jac = bad_grad
    options = dict(maxfun=20) if method == 'tnc' else dict(maxiter=20)
    if method in ['nelder-mead', 'powell', 'cobyla']:
        jac = None
    sol = optimize.minimize(func, x0, jac=jac, method=method, options=options)
    assert_equal(func(sol.x), sol.fun)
    if method == 'slsqp':
        pytest.xfail('SLSQP returns slightly worse')
    assert func(sol.x) <= f0