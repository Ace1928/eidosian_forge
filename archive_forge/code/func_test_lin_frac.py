import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_lin_frac(self) -> None:
    x = cp.Variable((2,), nonneg=True)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.arange(2)
    C = 2 * A
    d = np.arange(2)
    lin_frac = (cp.matmul(A, x) + b) / (cp.matmul(C, x) + d)
    self.assertTrue(lin_frac.is_dqcp())
    self.assertTrue(lin_frac.is_quasiconvex())
    self.assertTrue(lin_frac.is_quasiconcave())
    problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, lin_frac <= 1])
    self.assertTrue(problem.is_dqcp())
    problem.solve(SOLVER, qcp=True)
    self.assertAlmostEqual(problem.objective.value, 0, places=1)
    np.testing.assert_almost_equal(x.value, 0, decimal=5)