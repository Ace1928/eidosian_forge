import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_card_ls(self) -> None:
    n = 10
    np.random.seed(0)
    A = np.random.randn(n, n)
    x_star = np.random.randn(n)
    b = cp.matmul(A, x_star)
    epsilon = 0.001
    x = cp.Variable(n)
    objective_fn = cp.length(x)
    mse = cp.sum_squares(cp.matmul(A, x) - b) / n
    problem = cp.Problem(cp.Minimize(objective_fn), [mse <= epsilon])
    problem.solve(SOLVER, qcp=True)