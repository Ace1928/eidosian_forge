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
def test_float_is_invalid_index(self) -> None:
    with self.assertRaises(IndexError) as cm:
        self.x[1.0]
    self.assertEqual(str(cm.exception), 'float is an invalid index type.')
    with self.assertRaises(IndexError) as cm:
        self.x[1.0,]
    self.assertEqual(str(cm.exception), 'float is an invalid index type.')
    with self.assertRaises(IndexError) as cm:
        self.C[:2.0:40]
    self.assertEqual(str(cm.exception), 'float is an invalid index type.')
    with self.assertRaises(IndexError) as cm:
        self.x[np.array([1.0, 2.0])]
    self.assertEqual(str(cm.exception), 'arrays used as indices must be of integer (or boolean) type')