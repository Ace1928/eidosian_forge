from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def sum_expr(operators):
    """Add linear operators.

    Parameters
    ----------
    operators : list
        A list of linear operators.

    Returns
    -------
    LinOp
        A LinOp representing the sum of the operators.
    """
    return lo.LinOp(lo.SUM, operators[0].shape, operators, None)