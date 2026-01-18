from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def replace_new_vars(expr, id_to_new_var):
    """Replaces the given variables in the expression.

    Parameters
    ----------
    expr : LinOp
        The expression to replace variables in.
    id_to_new_var : dict
        A map of id to new variable.

    Returns
    -------
    LinOp
        An LinOp identical to expr, but with the given variables replaced.
    """
    if expr.type == lo.VARIABLE and expr.data in id_to_new_var:
        return id_to_new_var[expr.data]
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_new_vars(arg, id_to_new_var))
        return lo.LinOp(expr.type, expr.shape, new_args, expr.data)