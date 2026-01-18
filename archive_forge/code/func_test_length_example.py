import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_length_example(self) -> None:
    """Fix #1760."""
    n = 10
    np.random.seed(1)
    A = np.random.randn(n, n)
    x_star = np.random.randn(n)
    b = A @ x_star
    epsilon = 0.01
    x = cp.Variable(n)
    mse = cp.sum_squares(A @ x - b) / n
    problem = cp.Problem(cp.Minimize(cp.length(x)), [mse <= epsilon])
    assert problem.is_dqcp()
    problem.solve(qcp=True)
    assert np.isclose(problem.value, 8)