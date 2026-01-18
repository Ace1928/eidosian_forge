import numpy as np
import scipy.sparse as sp
from cvxpy.lin_ops.lin_op import (
from cvxpy.lin_ops.lin_utils import (
from cvxpy.tests.base_test import BaseTest
def test_neg_expr(self) -> None:
    """Test negating an expression.
        """
    shape = (5, 4)
    var = create_var(shape)
    expr = neg_expr(var)
    assert len(expr.args) == 1
    self.assertEqual(expr.shape, shape)
    self.assertEqual(expr.type, NEG)