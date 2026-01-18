from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def create_eq(lh_op, rh_op=None, constr_id=None):
    """Creates an internal equality constraint.

    Parameters
    ----------
    lh_term : LinOp
        The left-hand operator in the equality constraint.
    rh_term : LinOp
        The right-hand operator in the equality constraint.
    constr_id : int
        The id of the CVXPY equality constraint creating the constraint.

    Returns
    -------
    LinEqConstr
    """
    if constr_id is None:
        constr_id = get_id()
    expr = get_constr_expr(lh_op, rh_op)
    return LinEqConstr(expr, constr_id, lh_op.shape)