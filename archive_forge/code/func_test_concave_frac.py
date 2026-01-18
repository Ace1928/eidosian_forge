import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_concave_frac(self) -> None:
    x = cp.Variable(nonneg=True)
    concave_frac = cp.sqrt(x) / cp.exp(x)
    self.assertTrue(concave_frac.is_dqcp())
    self.assertTrue(concave_frac.is_quasiconcave())
    self.assertFalse(concave_frac.is_quasiconvex())
    problem = cp.Problem(cp.Maximize(concave_frac))
    self.assertTrue(problem.is_dqcp())
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, 0.428, places=1)
    self.assertAlmostEqual(x.value, 0.5, places=1)