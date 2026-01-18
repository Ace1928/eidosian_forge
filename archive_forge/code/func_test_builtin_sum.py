import numpy as np
import scipy.sparse as sp
import cvxpy
from cvxpy.tests.base_test import BaseTest
def test_builtin_sum(self) -> None:
    x = cvxpy.Variable(2, pos=True)
    self.assertTrue(sum(x).is_log_log_convex())