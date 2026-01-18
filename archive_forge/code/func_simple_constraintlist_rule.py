import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.deprecation import RenamedClass
from pyomo.common.errors import DeveloperError
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.expr import (
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.set import Set
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
def simple_constraintlist_rule(rule):
    """
    This is a decorator that translates None/True/False return values
    into ConstraintList.End/Constraint.Feasible/Constraint.Infeasible.
    This supports a simpler syntax in constraint rules, though these can be
    more difficult to debug when errors occur.

    Example use:

    @simple_constraintlist_rule
    def C_rule(model, i, j):
        ...

    model.c = ConstraintList(expr=simple_constraintlist_rule(...))
    """
    map_types = set([type(None)]) | native_logical_types
    result_map = {None: ConstraintList.End}
    for l_type in native_logical_types:
        result_map[l_type(True)] = Constraint.Feasible
        result_map[l_type(False)] = Constraint.Infeasible
    return rule_wrapper(rule, result_map, map_types=map_types)