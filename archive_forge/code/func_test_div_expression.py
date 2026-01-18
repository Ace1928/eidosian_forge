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
def test_div_expression(self) -> None:
    exp = self.x / 2
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.sign, s.UNKNOWN)
    self.assertEqual(exp.shape, (2,))
    with self.assertRaises(Exception) as cm:
        self.x / [2, 2, 3]
    print(cm.exception)
    self.assertRegex(str(cm.exception), 'Incompatible shapes for division.*')
    c = Constant([3.0, 4.0, 12.0])
    self.assertItemsAlmostEqual((c / Constant([1.0, 2.0, 3.0])).value, np.array([3.0, 2.0, 4.0]))
    c = Constant(2)
    exp = c / (3 - 5)
    self.assertEqual(exp.curvature, s.CONSTANT)
    self.assertEqual(exp.shape, tuple())
    self.assertEqual(exp.sign, s.NONPOS)
    p = Parameter(nonneg=True)
    exp = 2 / p
    p.value = 2
    self.assertEqual(exp.value, 1)
    rho = Parameter(nonneg=True)
    rho.value = 1
    self.assertEqual(rho.sign, s.NONNEG)
    self.assertEqual(Constant(2).sign, s.NONNEG)
    self.assertEqual((Constant(2) / Constant(2)).sign, s.NONNEG)
    self.assertEqual((Constant(2) * rho).sign, s.NONNEG)
    self.assertEqual((rho / 2).sign, s.NONNEG)
    x = cp.Variable((3, 3))
    c = np.arange(1, 4)[:, None]
    expr = x / c
    self.assertEqual((3, 3), expr.shape)
    x.value = np.ones((3, 3))
    A = np.ones((3, 3)) / c
    self.assertItemsAlmostEqual(A, expr.value)
    with self.assertRaises(Exception) as cm:
        x / c[:, 0]
    print(cm.exception)
    self.assertRegex(str(cm.exception), 'Incompatible shapes for division.*')