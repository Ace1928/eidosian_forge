from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def free_variables_in_active_equalities_set(blk):
    """
    Return a set of variables that are continued in active equalities.
    """
    vin = ComponentSet()
    for c in active_equalities(blk):
        for v in identify_variables(c.body):
            if not v.fixed:
                vin.add(v)
    return vin