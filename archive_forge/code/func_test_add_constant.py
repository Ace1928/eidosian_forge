import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_add_constant(self) -> None:
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.ceil(x) + 5), [x >= 2])
    problem.solve(SOLVER, qcp=True)
    np.testing.assert_almost_equal(x.value, 2)
    np.testing.assert_almost_equal(problem.objective.value, 7)