import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_infeasible_logistic_constr(self) -> None:
    x = cp.Variable(nonneg=True)
    constr = [cp.logistic(cp.ceil(x)) <= -5]
    problem = cp.Problem(cp.Minimize(0), constr)
    problem.solve(SOLVER, qcp=True)
    self.assertEqual(problem.status, s.INFEASIBLE)