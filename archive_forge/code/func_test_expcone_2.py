import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_expcone_2(self) -> None:
    x = cp.Variable(shape=(3,))
    tempcons = [cp.sum(x) <= 1.0, cp.sum(x) >= 0.1, x >= 0.01, cp.kl_div(x[1], x[0]) + x[1] - x[0] + x[2] <= 0]
    sigma = cp.suppfunc(x, tempcons)
    y = cp.Variable(shape=(3,))
    a = np.array([-3, -2, -1])
    expr = -sigma(y)
    objective = cp.Maximize(expr)
    cons = [y == a]
    prob = cp.Problem(objective, cons)
    prob.solve(solver='ECOS')
    epi_actual = prob.value
    direct_actual = expr.value
    expect = 0.235348211
    self.assertLessEqual(abs(epi_actual - expect), 1e-06)
    self.assertLessEqual(abs(direct_actual - expect), 1e-06)