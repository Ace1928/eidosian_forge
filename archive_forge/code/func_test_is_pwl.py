import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import (
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.linalg import gershgorin_psd_check
def test_is_pwl(self) -> None:
    """Test is_pwl()
        """
    A = np.ones((2, 3))
    b = np.ones(2)
    expr = A @ self.y - b
    self.assertEqual(expr.is_pwl(), True)
    expr = cp.maximum(1, 3 * self.y)
    self.assertEqual(expr.is_pwl(), True)
    expr = cp.abs(self.y)
    self.assertEqual(expr.is_pwl(), True)
    expr = cp.pnorm(3 * self.y, 1)
    self.assertEqual(expr.is_pwl(), True)
    expr = cp.pnorm(3 * self.y ** 2, 1)
    self.assertEqual(expr.is_pwl(), False)