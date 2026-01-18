import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_basic_ratio(self) -> None:
    x = cp.Variable()
    y = cp.Variable(nonneg=True)
    expr = x / y
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconcave())
    self.assertTrue(expr.is_quasiconvex())
    problem = cp.Problem(cp.Minimize(expr), [x == 12, y <= 6])
    self.assertTrue(problem.is_dqcp())
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, 2.0, places=1)
    self.assertAlmostEqual(x.value, 12, places=1)
    self.assertAlmostEqual(y.value, 6, places=1)
    x = cp.Variable()
    y = cp.Variable(nonpos=True)
    expr = x / y
    self.assertTrue(expr.is_dqcp())
    self.assertTrue(expr.is_quasiconcave())
    self.assertTrue(expr.is_quasiconvex())
    problem = cp.Problem(cp.Maximize(expr), [x == 12, y >= -6])
    self.assertTrue(problem.is_dqcp())
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, -2.0, places=1)
    self.assertAlmostEqual(x.value, 12, places=1)
    self.assertAlmostEqual(y.value, -6, places=1)