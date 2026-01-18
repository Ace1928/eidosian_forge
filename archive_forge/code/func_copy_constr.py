from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def copy_constr(constr, func):
    """Creates a copy of the constraint modified according to func.

    Parameters
    ----------
    constr : LinConstraint
        The constraint to modify.
    func : function
        Function to modify the constraint expression.

    Returns
    -------
    LinConstraint
        A copy of the constraint with the specified changes.
    """
    expr = func(constr.expr)
    return type(constr)(expr, constr.constr_id, constr.shape)