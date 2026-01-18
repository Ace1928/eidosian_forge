import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_lml(self) -> None:
    np.random.seed(0)
    k = 2
    x = cp.Parameter(4)
    y = cp.Variable(4)
    obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
    cons = [cp.sum(y) == k]
    problem = cp.Problem(cp.Minimize(obj), cons)
    x.value = np.array([1.0, -1.0, -1.0, -1.0])
    gradcheck(problem, solve_methods=[s.SCS], atol=0.01)
    perturbcheck(problem, solve_methods=[s.SCS], atol=0.0001)