import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_lp_7() -> SolverTestHelper:
    """Problem that takes significant time to solve - for testing time/iteration limits"""
    np.random.seed(0)
    n = 24 * 8
    c = cp.Variable((n,), pos=True)
    d = cp.Variable((n,), pos=True)
    c_or_d = cp.Variable((n,), boolean=True)
    big = 1000.0
    s = cp.cumsum(c * 0.9 - d)
    p = np.random.random(n)
    objective = cp.Maximize(p @ (d - c))
    constraints = [d <= 1, c <= 1, s >= 0, s <= 1, c <= c_or_d * big, d <= (1 - c_or_d) * big]
    return SolverTestHelper((objective, None), [(c, None), (d, None), (c_or_d, None)], [(con, None) for con in constraints])