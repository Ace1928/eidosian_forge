from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def rmul_expr(lh_op, rh_op, shape: Tuple[int, ...]):
    """Multiply two linear operators, with the constant on the right.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.
    shape : tuple
        The shape of the product.

    Returns
    -------
    LinOp
        A linear operator representing the product.
    """
    return lo.LinOp(lo.RMUL, shape, [lh_op], rh_op)