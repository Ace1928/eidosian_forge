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
def assert_exprs(self, *args):
    """Assert constraints as background axioms for the optimize solver."""
    args = _get_args(args)
    s = BoolSort(self.ctx)
    for arg in args:
        if isinstance(arg, Goal) or isinstance(arg, AstVector):
            for f in arg:
                Z3_optimize_assert(self.ctx.ref(), self.optimize, f.as_ast())
        else:
            arg = s.cast(arg)
            Z3_optimize_assert(self.ctx.ref(), self.optimize, arg.as_ast())