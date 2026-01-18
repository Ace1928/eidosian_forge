import numpy as np
import scipy.sparse as sp
from cvxpy.lin_ops.lin_op import (
from cvxpy.lin_ops.lin_utils import (
from cvxpy.tests.base_test import BaseTest
def test_add_expr(self) -> None:
    """Test adding lin expr.
        """
    shape = (5, 4)
    x = create_var(shape)
    y = create_var(shape)
    add_expr = sum_expr([x, y])
    self.assertEqual(add_expr.shape, shape)
    assert len(add_expr.args) == 2