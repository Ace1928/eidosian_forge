import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_concave_multiply(self) -> None:
    x, y = cp.Variable(2, nonneg=True)
    expr = cp.sqrt(x) * cp.sqrt(y)
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconcave())
    self.assertFalse(expr.is_quasiconvex())
    problem = cp.Problem(cp.Maximize(expr), [x <= 4, y <= 9])
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, 6, places=1)
    self.assertAlmostEqual(x.value, 4, places=1)
    self.assertAlmostEqual(y.value, 9, places=1)
    x, y = cp.Variable(2, nonneg=True)
    expr = (cp.sqrt(x) + 2.0) * (cp.sqrt(y) + 4.0)
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconcave())
    self.assertFalse(expr.is_quasiconvex())
    problem = cp.Problem(cp.Maximize(expr), [x <= 4, y <= 9])
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, 28, places=1)
    self.assertAlmostEqual(x.value, 4, places=1)
    self.assertAlmostEqual(y.value, 9, places=1)