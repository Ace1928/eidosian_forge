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
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('method', ['brent', 'bounded', 'golden'])
def test_nan_values(self, method):
    np.random.seed(1234)
    count = [0]

    def func(x):
        count[0] += 1
        if count[0] > 4:
            return np.nan
        else:
            return x ** 2 + 0.1 * np.sin(x)
    bracket = (-1, 0, 1)
    bounds = (-1, 1)
    with np.errstate(invalid='ignore'), suppress_warnings() as sup:
        sup.filter(UserWarning, 'delta_grad == 0.*')
        sup.filter(RuntimeWarning, '.*does not use Hessian.*')
        sup.filter(RuntimeWarning, '.*does not use gradient.*')
        count = [0]
        kwargs = {'bounds': bounds} if method == 'bounded' else {}
        sol = optimize.minimize_scalar(func, bracket=bracket, **kwargs, method=method, options=dict(maxiter=20))
        assert_equal(sol.success, False)