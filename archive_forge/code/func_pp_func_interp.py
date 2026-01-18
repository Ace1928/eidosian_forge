import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_func_interp(self, f):
    r = []
    sz = 0
    num = f.num_entries()
    for i in range(num):
        r.append(self.pp_func_entry(f.entry(i)))
        sz = sz + 1
        if sz > self.max_args:
            r.append(self.pp_ellipses())
            break
    if sz <= self.max_args:
        else_val = f.else_value()
        if else_val is None:
            else_pp = to_format('#unspecified')
        else:
            else_pp = self.pp_expr(else_val, 0, [])
        r.append(group(seq((to_format('else'), else_pp), self.pp_arrow())))
    return seq3(r, '[', ']')