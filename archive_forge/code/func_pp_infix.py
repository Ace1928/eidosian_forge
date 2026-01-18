import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_infix(self, a, d, xs):
    k = a.decl().kind()
    if self.is_infix_compact(k):
        op = self.pp_name(a)
        return group(seq(self.infix_args(a, d, xs), op, False))
    else:
        op = self.pp_name(a)
        sz = _len(op)
        op.string = ' ' + op.string
        op.size = sz + 1
        return group(seq(self.infix_args(a, d, xs), op))