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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('method', ['fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs', 'fmin_ncg', 'fmin_l_bfgs_b', 'fmin_tnc', 'fmin_slsqp'] + MINIMIZE_METHODS)
def test_minimize_callback_copies_array(self, method):
    if method in ('fmin_tnc', 'fmin_l_bfgs_b'):

        def func(x):
            return (optimize.rosen(x), optimize.rosen_der(x))
    else:
        func = optimize.rosen
        jac = optimize.rosen_der
        hess = optimize.rosen_hess
    x0 = np.zeros(10)
    kwargs = {}
    if method.startswith('fmin'):
        routine = getattr(optimize, method)
        if method == 'fmin_slsqp':
            kwargs['iter'] = 5
        elif method == 'fmin_tnc':
            kwargs['maxfun'] = 100
        elif method in ('fmin', 'fmin_powell'):
            kwargs['maxiter'] = 3500
        else:
            kwargs['maxiter'] = 5
    else:

        def routine(*a, **kw):
            kw['method'] = method
            return optimize.minimize(*a, **kw)
        if method == 'tnc':
            kwargs['options'] = dict(maxfun=100)
        else:
            kwargs['options'] = dict(maxiter=5)
    if method in ('fmin_ncg',):
        kwargs['fprime'] = jac
    elif method in ('newton-cg',):
        kwargs['jac'] = jac
    elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg', 'trust-constr'):
        kwargs['jac'] = jac
        kwargs['hess'] = hess
    results = []

    def callback(x, *args, **kwargs):
        assert not isinstance(x, optimize.OptimizeResult)
        results.append((x, np.copy(x)))
    routine(func, x0, callback=callback, **kwargs)
    assert len(results) > 2
    assert all((np.all(x == y) for x, y in results))
    combinations = itertools.combinations(results, 2)
    assert not any((np.may_share_memory(x[0], y[0]) for x, y in combinations))