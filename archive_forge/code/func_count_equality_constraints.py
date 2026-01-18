from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def count_equality_constraints(blk):
    """
    Count active equality constraints.
    """
    return len(active_equality_set(blk))