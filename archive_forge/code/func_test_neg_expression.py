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
def test_neg_expression(self) -> None:
    exp = -self.x
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (2,))
    assert exp.is_affine()
    self.assertEqual(exp.sign, s.UNKNOWN)
    assert not exp.is_nonneg()
    self.assertEqual(exp.shape, self.x.shape)
    exp = -self.C
    self.assertEqual(exp.curvature, s.AFFINE)
    self.assertEqual(exp.shape, (3, 2))