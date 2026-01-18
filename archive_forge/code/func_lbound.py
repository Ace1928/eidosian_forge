from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable
def lbound(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "lbound(...)"

    Parameters
    ==========

    array : Symbol or String
    dim : expr
    kind : expr

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import lbound
    >>> lb = lbound('arr', dim=2)
    >>> fcode(lb, source_format='free')
    'lbound(arr, 2)'

    """
    return FunctionCall('lbound', [_printable(array)] + ([_printable(dim)] if dim else []) + ([_printable(kind)] if kind else []))