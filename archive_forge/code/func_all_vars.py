from pyomo.common.collections import Bunch
from pyomo.core.base import Var, Constraint, Objective, maximize, minimize
from pyomo.repn.standard_repn import generate_standard_repn
def all_vars(b):
    """
        This conditionally chains together the active variables in the current block with
        the active variables in all of the parent blocks (if any exist).
        """
    for obj in b.component_objects(Var, active=True, descend_into=True):
        name = obj.parent_component().getname(fully_qualified=True, relative_to=b)
        yield (name, obj)
    b = b.parent_block()
    while not b is None:
        for obj in b.component_objects(Var, active=True, descend_into=False):
            name = obj.parent_component().name
            yield (name, obj)
        b = b.parent_block()