from pyomo.core import Var
from pyomo.core.base.indexed_component import UnindexedComponent_set
def replicate_var(comp, name, block, index_set=None):
    """
    Create a new variable that will have the same indexing set, domain,
    and bounds as the provided component, and add it to the given block.
    Optionally pass an index set to use that to build the variable, but
    this set must be symmetric to comp's index set.
    """
    new_var = create_var(comp, name, block, index_set)
    tighten_var_domain(comp, new_var, index_set)
    return new_var