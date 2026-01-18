from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def get_expr_vars(operator):
    """Get a list of the variables in the operator and their shapes.

    Parameters
    ----------
    operator : LinOp
        The operator to extract the variables from.

    Returns
    -------
    list
        A list of (var id, var shape) pairs.
    """
    if operator.type == lo.VARIABLE:
        return [(operator.data, operator.shape)]
    else:
        vars_ = []
        for arg in operator.args:
            vars_ += get_expr_vars(arg)
        return vars_