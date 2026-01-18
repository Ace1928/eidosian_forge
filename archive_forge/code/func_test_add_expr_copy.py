import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_add_expr_copy(self) -> None:
    """Test the copy function for AddExpresion class.
        """
    atom = self.x + self.y
    copy = atom.copy()
    self.assertTrue(type(copy) is type(atom))
    self.assertEqual(copy.args, atom.args)
    self.assertFalse(copy.args is atom.args)
    self.assertEqual(copy.get_data(), atom.get_data())
    copy = atom.copy(args=[self.A, self.B])
    self.assertTrue(type(copy) is type(atom))
    self.assertTrue(copy.args[0] is self.A)
    self.assertTrue(copy.args[1] is self.B)
    self.assertEqual(copy.get_data(), atom.get_data())