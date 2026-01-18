import numpy as np
import scipy.sparse as sp
from cvxpy.lin_ops.lin_op import (
from cvxpy.lin_ops.lin_utils import (
from cvxpy.tests.base_test import BaseTest
def test_eq_constr(self) -> None:
    """Test creating an equality constraint.
        """
    shape = (5, 5)
    x = create_var(shape)
    y = create_var(shape)
    lh_expr = sum_expr([x, y])
    value = np.ones(shape)
    rh_expr = create_const(value, shape)
    constr = create_eq(lh_expr, rh_expr)
    self.assertEqual(constr.shape, shape)
    vars_ = get_expr_vars(constr.expr)
    ref = [(x.data, shape), (y.data, shape)]
    self.assertCountEqual(vars_, ref)