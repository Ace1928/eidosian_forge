import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_basic_minimum(self) -> None:
    x, y = cp.Variable(2)
    expr = cp.minimum(cp.ceil(x), cp.ceil(y))
    problem = cp.Problem(cp.Maximize(expr), [x >= 11.9, x <= 15.8, y >= 17.4])
    self.assertTrue(problem.is_dqcp())
    problem.solve(SOLVER, qcp=True)
    self.assertEqual(problem.objective.value, 16.0)
    self.assertLess(x.value, 16.0)
    self.assertGreater(x.value, 14.9)
    self.assertGreater(y.value, 17.3)