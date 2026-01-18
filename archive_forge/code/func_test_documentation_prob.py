import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_documentation_prob(self) -> None:
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)
    a = cp.Parameter(pos=True, value=4.0)
    b = cp.Parameter(pos=True, value=2.0)
    c = cp.Parameter(pos=True, value=10.0)
    d = cp.Parameter(pos=True, value=1.0)
    objective_fn = x * y * z
    constraints = [a * x * y * z + b * x * z <= c, x <= b * y, y <= b * x, z >= d]
    problem = cp.Problem(cp.Maximize(objective_fn), constraints)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.01)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.01)