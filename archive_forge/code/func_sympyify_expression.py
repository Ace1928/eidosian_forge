import operator
import sys
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import value, native_types
def sympyify_expression(expr):
    """Convert a Pyomo expression to a Sympy expression"""
    object_map = PyomoSympyBimap()
    visitor = Pyomo2SympyVisitor(object_map)
    return (object_map, visitor.walk_expression(expr))