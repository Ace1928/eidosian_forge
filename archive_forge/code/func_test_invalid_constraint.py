import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_invalid_constraint(self) -> None:
    x = cp.Variable(shape=(3,))
    a = cp.Parameter(shape=(3,))
    cons = [a @ x == 1]
    with self.assertRaises(ValueError):
        cp.suppfunc(x, cons)