from pyomo.common.collections import ComponentMap
from pyomo.core import value
def disjunctive_ub(var, scope):
    """Compute the disjunctive upper bound for a variable in a given scope."""
    return disjunctive_bound(var, scope)[1]