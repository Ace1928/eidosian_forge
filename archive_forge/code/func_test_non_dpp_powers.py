import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_non_dpp_powers(self) -> None:
    s = cp.Parameter(1, nonneg=True)
    x = cp.Variable(1)
    obj = cp.Maximize(x + s)
    cons = [x <= 1]
    prob = cp.Problem(obj, cons)
    s.value = np.array([1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prob.solve(solver=cp.SCS, eps=1e-06)
    np.testing.assert_almost_equal(prob.value, 2.0, decimal=3)
    s = cp.Parameter(1, nonneg=True)
    x = cp.Variable(1)
    obj = cp.Maximize(x + s ** 2)
    cons = [x <= 1]
    prob = cp.Problem(obj, cons)
    s.value = np.array([1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prob.solve(solver=cp.SCS, eps=1e-06)
    np.testing.assert_almost_equal(prob.value, 2.0, decimal=3)
    s = cp.Parameter(1, nonneg=True)
    x = cp.Variable(1)
    obj = cp.Maximize(cp.multiply(x, s ** 2))
    cons = [x <= 1]
    prob = cp.Problem(obj, cons)
    s.value = np.array([1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prob.solve(solver=cp.SCS, eps=1e-06)
    np.testing.assert_almost_equal(prob.value, 1.0, decimal=3)