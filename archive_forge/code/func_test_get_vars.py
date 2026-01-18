import numpy as np
import scipy.sparse as sp
from cvxpy.lin_ops.lin_op import (
from cvxpy.lin_ops.lin_utils import (
from cvxpy.tests.base_test import BaseTest
def test_get_vars(self) -> None:
    """Test getting vars from an expression.
        """
    shape = (5, 4)
    x = create_var(shape)
    y = create_var(shape)
    A = create_const(np.ones(shape), shape)
    add_expr = sum_expr([x, y, A])
    vars_ = get_expr_vars(add_expr)
    ref = [(x.data, shape), (y.data, shape)]
    self.assertCountEqual(vars_, ref)