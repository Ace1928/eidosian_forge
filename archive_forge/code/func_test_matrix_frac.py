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
def test_matrix_frac(self) -> None:
    """Test for the matrix_frac atom.
        """
    atom = cp.matrix_frac(self.x, self.A)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    with self.assertRaises(Exception) as cm:
        cp.matrix_frac(self.x, self.C)
    self.assertEqual(str(cm.exception), 'The second argument to matrix_frac must be a square matrix.')
    with self.assertRaises(Exception) as cm:
        cp.matrix_frac(Variable(3), self.A)
    self.assertEqual(str(cm.exception), 'The arguments to matrix_frac have incompatible dimensions.')