import sys
import unittest
from typing import Tuple
import numpy
import scipy.sparse as sp
import cvxpy.interface.matrix_utilities as intf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
def assertExpression(self, expr, shape: Tuple[int, ...]) -> None:
    """Asserts that expr is an Expression with dimension shape.
        """
    assert isinstance(expr, Expression) or isinstance(expr, Constraint)
    self.assertEqual(expr.shape, shape)