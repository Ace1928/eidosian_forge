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
def substitute_vars(t, *m):
    """Substitute the free variables in t with the expression in m.

    >>> v0 = Var(0, IntSort())
    >>> v1 = Var(1, IntSort())
    >>> x  = Int('x')
    >>> f  = Function('f', IntSort(), IntSort(), IntSort())
    >>> # replace v0 with x+1 and v1 with x
    >>> substitute_vars(f(v0, v1), x + 1, x)
    f(x + 1, x)
    """
    if z3_debug():
        _z3_assert(is_expr(t), 'Z3 expression expected')
        _z3_assert(all([is_expr(n) for n in m]), 'Z3 invalid substitution, list of expressions expected.')
    num = len(m)
    _to = (Ast * num)()
    for i in range(num):
        _to[i] = m[i].as_ast()
    return _to_expr_ref(Z3_substitute_vars(t.ctx.ref(), t.as_ast(), num, _to), t.ctx)