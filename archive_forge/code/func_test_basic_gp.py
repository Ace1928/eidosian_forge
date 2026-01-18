import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_basic_gp(self) -> None:
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)
    a = cp.Parameter(pos=True)
    b = cp.Parameter(pos=True)
    c = cp.Parameter()
    constraints = [a * (x * y + x * z + y * z) <= b, x >= y ** c]
    problem = cp.Problem(cp.Minimize(1 / (x * y * z)), constraints)
    self.assertTrue(problem.is_dgp(dpp=True))
    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)