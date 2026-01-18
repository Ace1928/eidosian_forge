import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
def evaluate_fixed_subexpressions(expr, descend_into_named_expressions=True, remove_named_expressions=True):
    return EvaluateFixedSubexpressionVisitor(descend_into_named_expressions=descend_into_named_expressions, remove_named_expressions=remove_named_expressions).walk_expression(expr)