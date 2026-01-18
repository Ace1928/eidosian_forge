import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_pattern(self, a, d, xs):
    if a.num_args() == 1:
        return self.pp_expr(a.arg(0), d, xs)
    else:
        return seq1('MultiPattern', [self.pp_expr(arg, d + 1, xs) for arg in a.children()])