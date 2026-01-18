import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_l1_rectangle(self) -> None:
    np.random.seed(0)
    m, n = (3, 2)
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m, name='b')
    objective = cp.Minimize(cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective)
    self.assertTrue(problem.is_dpp())
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m)
    gradcheck(problem, atol=0.001)
    perturbcheck(problem, atol=0.001)