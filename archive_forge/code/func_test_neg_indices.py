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
def test_neg_indices(self) -> None:
    """Test negative indices.
        """
    c = Constant([[1, 2], [3, 4]])
    exp = c[-1, -1]
    self.assertEqual(exp.value, 4)
    self.assertEqual(exp.shape, tuple())
    self.assertEqual(exp.curvature, s.CONSTANT)
    c = Constant([1, 2, 3, 4])
    exp = c[1:-1]
    self.assertItemsAlmostEqual(exp.value, [2, 3])
    self.assertEqual(exp.shape, (2,))
    self.assertEqual(exp.curvature, s.CONSTANT)
    c = Constant([1, 2, 3, 4])
    exp = c[::-1]
    self.assertItemsAlmostEqual(exp.value, [4, 3, 2, 1])
    self.assertEqual(exp.shape, (4,))
    self.assertEqual(exp.curvature, s.CONSTANT)
    x = Variable(4)
    self.assertEqual(x[::-1].shape, (4,))
    Problem(Minimize(0), [x[::-1] == c]).solve(solver=cp.SCS)
    self.assertItemsAlmostEqual(x.value, [4, 3, 2, 1])
    x = Variable(2)
    self.assertEqual(x[::-1].shape, (2,))
    x = Variable(100, name='x')
    self.assertEqual('x[0:99]', str(x[:-1]))
    c = Constant([[1, 2], [3, 4]])
    expr = c[0, 2:0:-1]
    self.assertEqual(expr.shape, (1,))
    self.assertAlmostEqual(expr.value, 3)
    expr = c[0, 2::-1]
    self.assertEqual(expr.shape, (2,))
    self.assertItemsAlmostEqual(expr.value, [3, 1])