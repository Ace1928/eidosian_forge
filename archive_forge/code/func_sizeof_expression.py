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
def sizeof_expression(expr):
    """
    Return the number of nodes in the expression tree.

    Args:
        expr: The root node of an expression tree.

    Returns:
        A non-negative integer that is the number of
        interior and leaf nodes in the expression tree.
    """

    def enter(node):
        return (None, 1)

    def accept(node, data, child_result, child_idx):
        return data + child_result
    return StreamBasedExpressionVisitor(enterNode=enter, acceptChildResult=accept).walk_expression(expr)