from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def degrees_of_freedom(blk):
    """
    Return the degrees of freedom.

    Args:
        blk (Block or _BlockData): Block to count degrees of freedom in
    Returns:
        (int): Number of degrees of freedom
    """
    return count_free_variables(blk) - count_equality_constraints(blk)