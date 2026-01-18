from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def replace_params_with_consts(expr):
    """Replaces parameters with constant nodes.

    Parameters
    ----------
    expr : LinOp
        The expression to replace parameters in.

    Returns
    -------
    LinOp
        An LinOp identical to expr, but with the parameters replaced.
    """
    if expr.type == lo.PARAM:
        return create_const(check_param_val(expr.data), expr.shape)
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_params_with_consts(arg))
        if isinstance(expr.data, lo.LinOp) and expr.data.type == lo.PARAM:
            data_lin_op = expr.data
            assert isinstance(data_lin_op.shape, tuple)
            val = check_param_val(data_lin_op.data)
            data = create_const(val, data_lin_op.shape)
        else:
            data = expr.data
        return lo.LinOp(expr.type, expr.shape, new_args, data)