import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_var(self, a, d, xs):
    idx = z3.get_var_index(a)
    sz = len(xs)
    if idx >= sz:
        return to_format('&#957;<sub>%s</sub>' % idx, 1)
    else:
        return to_format(xs[sz - idx - 1])