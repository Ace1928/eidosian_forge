import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_invalid_solver(self) -> None:
    n = 3
    x = cp.Variable(shape=(n,))
    sigma = cp.suppfunc(x, [cp.norm(x - np.random.randn(n), 2) <= 1])
    y_var = cp.Variable(shape=(n,))
    prob = cp.Problem(cp.Minimize(sigma(y_var)), [np.random.randn(n) == y_var])
    with self.assertRaisesRegex(SolverError, '.*could not be reduced to a QP.*'):
        prob.solve(solver='OSQP')