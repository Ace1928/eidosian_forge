import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_dist_ratio(self) -> None:
    x = cp.Variable(2)
    a = np.ones(2)
    b = np.zeros(2)
    problem = cp.Problem(cp.Minimize(cp.dist_ratio(x, a, b)), [x <= 0.8])
    problem.solve(SOLVER, qcp=True)
    np.testing.assert_almost_equal(problem.objective.value, 0.25)
    np.testing.assert_almost_equal(x.value, np.array([0.8, 0.8]))