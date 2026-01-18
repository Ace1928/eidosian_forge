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
def identify_variables(expr, include_fixed=True):
    """
    A generator that yields a sequence of variables
    in an expression tree.

    Args:
        expr: The root node of an expression tree.
        include_fixed (bool): If :const:`True`, then
            this generator will yield variables whose
            value is fixed.  Defaults to :const:`True`.

    Yields:
        Each variable that is found.
    """
    visitor = _VariableVisitor()
    if include_fixed:
        for v in visitor.xbfs_yield_leaves(expr):
            if isinstance(v, tuple):
                yield from v
            else:
                yield v
    else:
        for v in visitor.xbfs_yield_leaves(expr):
            if isinstance(v, tuple):
                for v_i in v:
                    if not v_i.is_fixed():
                        yield v_i
            elif not v.is_fixed():
                yield v