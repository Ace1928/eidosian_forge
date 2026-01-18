from pyomo.common.collections import ComponentMap
from pyomo.core import value
def disjunctive_bound(var, scope):
    """Compute the disjunctive bounds for a variable in a given scope.

    Args:
        var (_VarData): Variable for which to compute bound
        scope (Component): The scope in which to compute the bound. If not a
            _DisjunctData, it will walk up the tree and use the scope of the
            most immediate enclosing _DisjunctData.

    Returns:
        numeric: the tighter of either the disjunctive lower bound, the
            variable lower bound, or (-inf, inf) if neither exist.

    """
    var_bnd = (value(var.lb) if var.has_lb() else -inf, value(var.ub) if var.has_ub() else inf)
    possible_disjunct = scope
    while possible_disjunct is not None:
        try:
            disj_bnd = possible_disjunct._disj_var_bounds.get(var, (-inf, inf))
            disj_bnd = (max(var_bnd[0], disj_bnd[0]), min(var_bnd[1], disj_bnd[1]))
            return disj_bnd
        except AttributeError:
            possible_disjunct = possible_disjunct.parent_block()
    return var_bnd