from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def div_expr(lh_op, rh_op):
    """Divide one linear operator by another.

    Assumes rh_op is a scalar constant.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the quotient.
    rh_op : LinOp
        The right-hand operator in the quotient.
    shape : tuple
        The shape of the quotient.

    Returns
    -------
    LinOp
        A linear operator representing the quotient.
    """
    return lo.LinOp(lo.DIV, lh_op.shape, [lh_op], rh_op)