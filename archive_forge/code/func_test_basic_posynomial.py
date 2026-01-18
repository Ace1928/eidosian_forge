import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_basic_posynomial(self) -> None:
    alpha = cp.Parameter(pos=True, value=1.0)
    beta = cp.Parameter(pos=True, value=2.0)
    kappa = cp.Parameter(pos=True, value=3.0)
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    monomial_one = alpha * beta * x
    monomial_two = beta * kappa * x * y
    posynomial = monomial_one + monomial_two
    problem = cp.Problem(cp.Minimize(posynomial), [x == alpha, y == beta])
    self.assertTrue(problem.is_dgp())
    self.assertTrue(problem.is_dgp(dpp=True))
    self.assertFalse(problem.is_dpp('dcp'))
    problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    self.assertAlmostEqual(x.value, 1.0)
    self.assertAlmostEqual(y.value, 2.0)
    self.assertAlmostEqual(problem.value, 14.0, places=3)
    alpha.value = 4.0
    beta.value = 5.0
    problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    self.assertAlmostEqual(x.value, 4.0)
    self.assertAlmostEqual(y.value, 5.0)
    self.assertAlmostEqual(problem.value, 380.0, places=3)