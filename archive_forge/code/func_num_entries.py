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
def num_entries(self):
    """Return the number of entries/points in the function interpretation `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0) == 1, f(1) == 1, f(2) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[f]
        [2 -> 0, else -> 1]
        >>> m[f].num_entries()
        1
        """
    return int(Z3_func_interp_get_num_entries(self.ctx.ref(), self.f))