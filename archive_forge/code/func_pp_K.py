import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_K(self, a, d, xs):
    return seq1(self.pp_name(a), [self.pp_sort(a.domain()), self.pp_expr(a.arg(0), d + 1, xs)])