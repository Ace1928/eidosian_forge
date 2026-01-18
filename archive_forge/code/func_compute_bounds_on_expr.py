from collections import defaultdict
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, value
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
import logging
from pyomo.common.errors import InfeasibleConstraintException, PyomoException
from pyomo.common.config import (
from pyomo.common.numeric_types import native_types
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
def compute_bounds_on_expr(expr, ignore_fixed=False):
    """
    Compute bounds on an expression based on the bounds on the variables in
    the expression.

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.NumericExpression
    ignore_fixed: bool, treats fixed Vars as constants if False, else treats
                  them as Vars

    Returns
    -------
    lb: float
    ub: float
    """
    lb, ub = ExpressionBoundsVisitor(use_fixed_var_values_as_bounds=not ignore_fixed).walk_expression(expr)
    if lb == -interval.inf:
        lb = None
    if ub == interval.inf:
        ub = None
    return (lb, ub)