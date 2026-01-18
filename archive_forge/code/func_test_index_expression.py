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
def test_index_expression(self) -> None:
    exp = self.x[1]
    self.assertEqual(exp.curvature, s.AFFINE)
    assert exp.is_affine()
    self.assertEqual(exp.shape, tuple())
    self.assertEqual(exp.value, None)
    exp = self.x[1].T
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, tuple())
    with self.assertRaises(Exception) as cm:
        self.x[2, 0]
    self.assertEqual(str(cm.exception), 'Too many indices for expression.')
    with self.assertRaises(Exception) as cm:
        self.x[2]
    self.assertEqual(str(cm.exception), 'Index 2 is out of bounds for axis 0 with size 2.')
    exp = self.C[0:2, 1]
    self.assertEqual(exp.shape, (2,))
    exp = self.C[0:, 0:2]
    self.assertEqual(exp.shape, (3, 2))
    exp = self.C[0::2, 0::2]
    self.assertEqual(exp.shape, (2, 1))
    exp = self.C[:3, :1:2]
    self.assertEqual(exp.shape, (3, 1))
    exp = self.C[0:, 0]
    self.assertEqual(exp.shape, (3,))
    c = Constant([[1, -2], [0, 4]])
    exp = c[1, 1]
    self.assertEqual(exp.curvature, s.CONSTANT)
    self.assertEqual(exp.sign, s.UNKNOWN)
    self.assertEqual(c[0, 1].sign, s.UNKNOWN)
    self.assertEqual(c[1, 0].sign, s.UNKNOWN)
    self.assertEqual(exp.shape, tuple())
    self.assertEqual(exp.value, 4)
    c = Constant([[1, -2, 3], [0, 4, 5], [7, 8, 9]])
    exp = c[0:3, 0:4:2]
    self.assertEqual(exp.curvature, s.CONSTANT)
    assert exp.is_constant()
    self.assertEqual(exp.shape, (3, 2))
    self.assertEqual(exp[0, 1].value, 7)
    exp = self.C.T[0:2, 1:2]
    self.assertEqual(exp.shape, (2, 1))
    exp = (self.x + self.z)[1]
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.sign, s.UNKNOWN)
    self.assertEqual(exp.shape, tuple())
    exp = (self.x + self.a)[1:2]
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (1,))
    exp = (self.x - self.z)[1:2]
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (1,))
    exp = (self.x - self.a)[1]
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, tuple())
    exp = (-self.x)[1]
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, tuple())
    c = Constant([[1, 2], [3, 4]])
    exp = (c @ self.x)[1]
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, tuple())
    c = Constant([[1, 2], [3, 4]])
    exp = (c * self.a)[1, 0:1]
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (1,))