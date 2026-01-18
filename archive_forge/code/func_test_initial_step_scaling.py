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
def test_initial_step_scaling(self):
    scales = [1e-50, 1, 1e+50]
    methods = ['CG', 'BFGS', 'L-BFGS-B', 'Newton-CG']

    def f(x):
        if first_step_size[0] is None and x[0] != x0[0]:
            first_step_size[0] = abs(x[0] - x0[0])
        if abs(x).max() > 10000.0:
            raise AssertionError('Optimization stepped far away!')
        return scale * (x[0] - 1) ** 2

    def g(x):
        return np.array([scale * (x[0] - 1)])
    for scale, method in itertools.product(scales, methods):
        if method in ('CG', 'BFGS'):
            options = dict(gtol=scale * 1e-08)
        else:
            options = dict()
        if scale < 1e-10 and method in ('L-BFGS-B', 'Newton-CG'):
            continue
        x0 = [-1.0]
        first_step_size = [None]
        res = optimize.minimize(f, x0, jac=g, method=method, options=options)
        err_msg = f'{method} {scale}: {first_step_size}: {res}'
        assert res.success, err_msg
        assert_allclose(res.x, [1.0], err_msg=err_msg)
        assert res.nit <= 3, err_msg
        if scale > 1e-10:
            if method in ('CG', 'BFGS'):
                assert_allclose(first_step_size[0], 1.01, err_msg=err_msg)
            else:
                assert first_step_size[0] > 0.5 and first_step_size[0] < 3, err_msg
        else:
            pass