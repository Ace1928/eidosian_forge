import collections
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from pyomo.core.expr import (
from typing import List
from pyomo.common.collections import Sequence
from pyomo.common.errors import PyomoException
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import native_types
def compare_expressions(expr1, expr2, include_named_exprs=True):
    """Returns True if 2 expression trees are identical, False otherwise.

    Parameters
    ----------
    expr1: NumericValue
        A Pyomo Var, Param, or expression
    expr2: NumericValue
        A Pyomo Var, Param, or expression
    include_named_exprs: bool
        If False, then named expressions will be ignored. In other
        words, this function will return True if one expression has a
        named expression and the other does not as long as the rest of
        the expression trees are identical.

    Returns
    -------
    res: bool
        A bool indicating whether or not the expressions are identical.

    """
    pn1 = convert_expression_to_prefix_notation(expr1, include_named_exprs=include_named_exprs)
    pn2 = convert_expression_to_prefix_notation(expr2, include_named_exprs=include_named_exprs)
    try:
        res = pn1 == pn2
    except PyomoException:
        res = False
    return res