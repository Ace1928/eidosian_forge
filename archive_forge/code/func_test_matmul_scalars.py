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
def test_matmul_scalars(self) -> None:
    """Test evaluating a matmul that reduces one argument internally to a scalar.
        """
    x = cp.Variable((2,))
    quad = cp.quad_form(x, np.eye(2))
    a = np.array([2])
    expr = quad * a
    x.value = np.array([1, 2])
    P = np.eye(2)
    true_val = np.transpose(x.value) @ P @ x.value * a
    assert quad.shape == ()
    self.assertEqual(expr.value, true_val)