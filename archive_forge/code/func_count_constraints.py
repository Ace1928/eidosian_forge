from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def count_constraints(blk):
    """
    Count active constraints.
    """
    return len(active_constraint_set(blk))