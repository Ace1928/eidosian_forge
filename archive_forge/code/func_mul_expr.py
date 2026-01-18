from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def mul_expr(lh_op, rh_op, shape: Tuple[int, ...]):
    """Multiply two linear operators, with the constant on the left.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.

    Returns
    -------
    LinOp
        A linear operator representing the product.
    """
    return lo.LinOp(lo.MUL, shape, [rh_op], lh_op)