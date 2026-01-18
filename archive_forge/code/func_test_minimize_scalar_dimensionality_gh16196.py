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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('method', MINIMIZE_SCALAR_METHODS)
@pytest.mark.parametrize('tol', [1, 1e-06])
@pytest.mark.parametrize('fshape', [(), (1,), (1, 1)])
def test_minimize_scalar_dimensionality_gh16196(self, method, tol, fshape):

    def f(x):
        return np.array(x ** 4).reshape(fshape)
    a, b = (-0.1, 0.2)
    kwargs = dict(bracket=(a, b)) if method != 'bounded' else dict(bounds=(a, b))
    kwargs.update(dict(method=method, tol=tol))
    res = optimize.minimize_scalar(f, **kwargs)
    assert res.x.shape == res.fun.shape == f(res.x).shape == fshape