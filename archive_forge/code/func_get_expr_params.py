from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def get_expr_params(operator):
    """Get a list of the parameters in the operator.

    Parameters
    ----------
    operator : LinOp
        The operator to extract the parameters from.

    Returns
    -------
    list
        A list of parameter objects.
    """
    if operator.type == lo.PARAM:
        return operator.data.parameters()
    else:
        params = []
        for arg in operator.args:
            params += get_expr_params(arg)
        if isinstance(operator.data, lo.LinOp):
            params += get_expr_params(operator.data)
        return params