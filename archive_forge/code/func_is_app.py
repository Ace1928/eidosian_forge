from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def is_app(a):
    """Return `True` if `a` is a Z3 function application.

    Note that, constants are function applications with 0 arguments.

    >>> a = Int('a')
    >>> is_app(a)
    True
    >>> is_app(a + 1)
    True
    >>> is_app(IntSort())
    False
    >>> is_app(1)
    False
    >>> is_app(IntVal(1))
    True
    >>> x = Int('x')
    >>> is_app(ForAll(x, x >= 0))
    False
    """
    if not isinstance(a, ExprRef):
        return False
    k = _ast_kind(a.ctx, a)
    return k == Z3_NUMERAL_AST or k == Z3_APP_AST