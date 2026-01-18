import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_basic_monomial(self) -> None:
    alpha = cp.Parameter(pos=True, value=1.0)
    beta = cp.Parameter(pos=True, value=2.0)
    x = cp.Variable(pos=True)
    monomial = alpha * beta * x
    problem = cp.Problem(cp.Minimize(monomial), [x == alpha])
    self.assertTrue(problem.is_dgp())
    self.assertTrue(problem.is_dgp(dpp=True))
    self.assertFalse(problem.is_dpp('dcp'))
    problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    self.assertAlmostEqual(x.value, 1.0)
    self.assertAlmostEqual(problem.value, 2.0)
    alpha.value = 3.0
    problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
    self.assertAlmostEqual(x.value, 3.0)
    self.assertAlmostEqual(problem.value, 18.0)