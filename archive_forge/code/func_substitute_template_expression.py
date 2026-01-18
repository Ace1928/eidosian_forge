import itertools
import logging
import sys
import builtins
from contextlib import nullcontext
from pyomo.common.errors import TemplateExpressionError
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
def substitute_template_expression(expr, substituter, *args, **kwargs):
    """Substitute IndexTemplates in an expression tree.

    This is a general utility function for walking the expression tree
    and substituting all occurrences of IndexTemplate and
    GetItemExpression nodes.

    Args:
        substituter: method taking (expression, *args) and returning
           the new object
        *args: these are passed directly to the substituter

    Returns:
        a new expression tree with all substitutions done
    """
    visitor = ReplaceTemplateExpression(substituter, *args, **kwargs)
    return visitor.walk_expression(expr)