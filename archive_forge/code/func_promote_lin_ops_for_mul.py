from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def promote_lin_ops_for_mul(lh_op, rh_op):
    """Promote arguments for multiplication.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the multiplication.
    rh_op : LinOp
        The right-hand operator in the multiplication.

    Returns
    -------
    LinOp
       Promoted left-hand operator.
    LinOp
       Promoted right-hand operator.
    tuple
       Shape of the product
    """
    lh_shape, rh_shape, shape = u.shape.mul_shapes_promote(lh_op.shape, rh_op.shape)
    lh_op = lo.LinOp(lh_op.type, lh_shape, lh_op.args, lh_op.data)
    rh_op = lo.LinOp(rh_op.type, rh_shape, rh_op.args, rh_op.data)
    return (lh_op, rh_op, shape)