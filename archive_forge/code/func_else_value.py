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
def else_value(self):
    """
        Return the `else` value for a function interpretation.
        Return None if Z3 did not specify the `else` value for
        this object.

        >>> f = Function('f', IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0) == 1, f(1) == 1, f(2) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[f]
        [2 -> 0, else -> 1]
        >>> m[f].else_value()
        1
        """
    r = Z3_func_interp_get_else(self.ctx.ref(), self.f)
    if r:
        return _to_expr_ref(r, self.ctx)
    else:
        return None