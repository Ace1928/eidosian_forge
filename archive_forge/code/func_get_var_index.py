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
def get_var_index(a):
    """Return the de-Bruijn index of the Z3 bounded variable `a`.

    >>> x = Int('x')
    >>> y = Int('y')
    >>> is_var(x)
    False
    >>> is_const(x)
    True
    >>> f = Function('f', IntSort(), IntSort(), IntSort())
    >>> # Z3 replaces x and y with bound variables when ForAll is executed.
    >>> q = ForAll([x, y], f(x, y) == x + y)
    >>> q.body()
    f(Var(1), Var(0)) == Var(1) + Var(0)
    >>> b = q.body()
    >>> b.arg(0)
    f(Var(1), Var(0))
    >>> v1 = b.arg(0).arg(0)
    >>> v2 = b.arg(0).arg(1)
    >>> v1
    Var(1)
    >>> v2
    Var(0)
    >>> get_var_index(v1)
    1
    >>> get_var_index(v2)
    0
    """
    if z3_debug():
        _z3_assert(is_var(a), 'Z3 bound variable expected')
    return int(Z3_get_index_value(a.ctx.ref(), a.as_ast()))