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
def asoft(a):
    v = Z3_optimize_assert_soft(self.ctx.ref(), self.optimize, a.as_ast(), weight, id)
    return OptimizeObjective(self, v, False)