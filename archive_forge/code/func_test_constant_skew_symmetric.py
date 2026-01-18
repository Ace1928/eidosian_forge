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
def test_constant_skew_symmetric(self) -> None:
    M1_false = np.eye(3)
    M2_true = np.zeros((3, 3))
    M3_true = np.array([[0, 1], [-1, 0]])
    M4_true = np.array([[0, -1], [1, 0]])
    M5_false = np.array([[0, 1], [1, 0]])
    M6_false = np.array([[1, 1], [-1, 0]])
    M7_false = np.array([[0, 1], [-1.1, 0]])
    C = Constant(M1_false)
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(M2_true)
    self.assertTrue(C.is_skew_symmetric())
    C = Constant(M3_true)
    self.assertTrue(C.is_skew_symmetric())
    C = Constant(M4_true)
    self.assertTrue(C.is_skew_symmetric())
    C = Constant(M5_false)
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(M6_false)
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(M7_false)
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(sp.csc_matrix(M1_false))
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(sp.csc_matrix(M2_true))
    self.assertTrue(C.is_skew_symmetric())
    C = Constant(sp.csc_matrix(M4_true))
    self.assertTrue(C.is_skew_symmetric())
    C = Constant(sp.csc_matrix(M5_false))
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(sp.csc_matrix(M6_false))
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(sp.csc_matrix(M7_false))
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(1j * M2_true)
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(1j * M3_true)
    self.assertFalse(C.is_skew_symmetric())
    C = Constant(1j * M4_true)
    self.assertFalse(C.is_skew_symmetric())
    pass