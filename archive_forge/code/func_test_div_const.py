import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_div_const(self) -> None:
    x = cp.Variable()
    obj = cp.Minimize(cp.ceil(x) / 0.5)
    problem = cp.Problem(obj, [x >= 10])
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(x.value, 10, places=1)
    self.assertAlmostEqual(problem.value, 20, places=1)
    x = cp.Variable()
    obj = cp.Maximize(cp.ceil(x) / -0.5)
    problem = cp.Problem(obj, [x >= 10])
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(x.value, 10, places=1)
    self.assertAlmostEqual(problem.value, -20, places=1)