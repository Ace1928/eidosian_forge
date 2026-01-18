from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def diag_mat(operator, k: int=0):
    """Converts the diagonal of a matrix to a vector.

    Parameters
    ----------
    operator : LinOp
        The operator to convert to a vector.
    k : int
        The offset of the diagonal.

    Returns
    -------
    LinOp
       LinOp representing the matrix diagonal.
    """
    shape = (operator.shape[0] - abs(k), 1)
    return lo.LinOp(lo.DIAG_MAT, shape, [operator], k)