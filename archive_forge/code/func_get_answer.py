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
def get_answer(self):
    """Retrieve answer from last query call."""
    r = Z3_fixedpoint_get_answer(self.ctx.ref(), self.fixedpoint)
    return _to_expr_ref(r, self.ctx)