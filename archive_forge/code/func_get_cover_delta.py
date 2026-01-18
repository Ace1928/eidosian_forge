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
def get_cover_delta(self, level, predicate):
    """Retrieve properties known about predicate for the level'th unfolding.
        -1 is treated as the limit (infinity)
        """
    r = Z3_fixedpoint_get_cover_delta(self.ctx.ref(), self.fixedpoint, level, predicate.ast)
    return _to_expr_ref(r, self.ctx)