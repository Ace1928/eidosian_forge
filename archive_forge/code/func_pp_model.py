import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_model(self, m):
    r = []
    sz = 0
    for d in m:
        i = m[d]
        if isinstance(i, z3.FuncInterp):
            i_pp = self.pp_func_interp(i)
        else:
            i_pp = self.pp_expr(i, 0, [])
        name = self.pp_name(d)
        r.append(compose(name, to_format(' = '), indent(_len(name) + 3, i_pp)))
        sz = sz + 1
        if sz > self.max_args:
            r.append(self.pp_ellipses())
            break
    return seq3(r, '[', ']')