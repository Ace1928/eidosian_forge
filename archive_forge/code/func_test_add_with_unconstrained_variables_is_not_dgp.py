import numpy as np
import scipy.sparse as sp
import cvxpy
from cvxpy.tests.base_test import BaseTest
def test_add_with_unconstrained_variables_is_not_dgp(self) -> None:
    x = cvxpy.Variable()
    y = cvxpy.Variable(pos=True)
    expr = x + y
    self.assertTrue(not expr.is_dgp())
    self.assertTrue(not expr.is_log_log_convex())
    self.assertTrue(not expr.is_log_log_concave())
    posynomial = 5.0 * x * y + 1.2 * y * y
    self.assertTrue(not posynomial.is_dgp())
    self.assertTrue(not posynomial.is_log_log_convex())
    self.assertTrue(not posynomial.is_log_log_concave())