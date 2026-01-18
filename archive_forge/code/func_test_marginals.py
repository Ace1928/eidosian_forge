import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def test_marginals(self):
    c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=0)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
    lb, ub = bounds.T

    def f_bub(x):
        return linprog(c, A_ub, x, A_eq, b_eq, bounds, method=self.method).fun
    dfdbub = approx_derivative(f_bub, b_ub, method='3-point', f0=res.fun)
    assert_allclose(res.ineqlin.marginals, dfdbub)

    def f_beq(x):
        return linprog(c, A_ub, b_ub, A_eq, x, bounds, method=self.method).fun
    dfdbeq = approx_derivative(f_beq, b_eq, method='3-point', f0=res.fun)
    assert_allclose(res.eqlin.marginals, dfdbeq)

    def f_lb(x):
        bounds = np.array([x, ub]).T
        return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method).fun
    with np.errstate(invalid='ignore'):
        dfdlb = approx_derivative(f_lb, lb, method='3-point', f0=res.fun)
        dfdlb[~np.isfinite(lb)] = 0
    assert_allclose(res.lower.marginals, dfdlb)

    def f_ub(x):
        bounds = np.array([lb, x]).T
        return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method).fun
    with np.errstate(invalid='ignore'):
        dfdub = approx_derivative(f_ub, ub, method='3-point', f0=res.fun)
        dfdub[~np.isfinite(ub)] = 0
    assert_allclose(res.upper.marginals, dfdub)