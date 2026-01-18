from pyomo.core.expr.numeric_expr import LinearExpression
import pyomo.environ as pyo
from pyomo.core import Objective
def find_active_objective(pyomomodel):
    obj = list(pyomomodel.component_data_objects(Objective, active=True, descend_into=True))
    if len(obj) != 1:
        raise RuntimeError("Could not identify exactly one active Objective for model '%s' (found %d objectives)" % (pyomomodel.name, len(obj)))
    return obj[0]