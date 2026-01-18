from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
def find_infeasible_constraints(m, tol=1e-06):
    """Find the infeasible constraints in the model.

    Uses the current model state.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    Yields
    ------
    constr: ConstraintData
        The infeasible constraint object

    body_value: float or None
        The numeric value of the constraint body (or None if there was an
        error evaluating the expression)

    infeasible: int
        A bitmask indicating which bound was infeasible (1 for the lower
        bound, 2 for the upper bound, or 4 if the body or bound was
        undefined)

    """
    for constr in m.component_data_objects(ctype=Constraint, active=True, descend_into=True):
        body_value = value(constr.body, exception=False)
        infeasible = _check_infeasible(constr, body_value, tol)
        if infeasible:
            yield (constr, body_value, infeasible)