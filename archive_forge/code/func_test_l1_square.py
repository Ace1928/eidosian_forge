import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_l1_square(self) -> None:
    np.random.seed(0)
    n = 3
    x = cp.Variable(n)
    A = cp.Parameter((n, n))
    b = cp.Parameter(n, name='b')
    objective = cp.Minimize(cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective)
    self.assertTrue(problem.is_dpp())
    L = np.random.randn(n, n)
    A.value = L.T @ L + np.eye(n)
    b.value = np.random.randn(n)
    gradcheck(problem)
    perturbcheck(problem)