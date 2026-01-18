from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def reverse_sd(expr):
    """
    First order reverse symbolic differentiation

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.NumericExpression
        expression to differentiate

    Returns
    -------
    ComponentMap
        component_map mapping variables to derivatives with respect
        to the corresponding variable
    """
    return _reverse_diff_helper(expr, False)