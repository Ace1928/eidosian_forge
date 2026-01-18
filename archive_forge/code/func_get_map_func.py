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
def get_map_func(a):
    """Return the function declaration associated with a Z3 map array expression.

    >>> f = Function('f', IntSort(), IntSort())
    >>> b = Array('b', IntSort(), IntSort())
    >>> a  = Map(f, b)
    >>> eq(f, get_map_func(a))
    True
    >>> get_map_func(a)
    f
    >>> get_map_func(a)(0)
    f(0)
    """
    if z3_debug():
        _z3_assert(is_map(a), 'Z3 array map expression expected.')
    return FuncDeclRef(Z3_to_func_decl(a.ctx_ref(), Z3_get_decl_ast_parameter(a.ctx_ref(), a.decl().ast, 0)), ctx=a.ctx)