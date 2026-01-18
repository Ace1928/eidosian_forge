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
def substitute_template_with_value(expr):
    """A simple substituter to expand expression for current template

    This substituter will replace all GetItemExpression / IndexTemplate
    nodes with the actual _ComponentData based on the current value of
    the IndexTemplate(s)

    """
    if type(expr) is IndexTemplate:
        return as_numeric(expr())
    else:
        return resolve_template(expr)