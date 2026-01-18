import operator
import sys
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import value, native_types
def sympy2pyomo_expression(expr, object_map):
    visitor = Sympy2PyomoVisitor(object_map)
    return visitor.walk_expression(expr)