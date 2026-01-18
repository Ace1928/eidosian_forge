from pyomo.common.collections import ComponentMap
from pyomo.core import value
def disjunctive_bounds(scope):
    """Return all of the variable bounds defined at a disjunctive scope."""
    possible_disjunct = scope
    while possible_disjunct is not None:
        try:
            return possible_disjunct._disj_var_bounds
        except AttributeError:
            possible_disjunct = possible_disjunct.parent_block()
    return ComponentMap()