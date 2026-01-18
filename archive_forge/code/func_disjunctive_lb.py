from pyomo.common.collections import ComponentMap
from pyomo.core import value
def disjunctive_lb(var, scope):
    """Compute the disjunctive lower bound for a variable in a given scope."""
    return disjunctive_bound(var, scope)[0]