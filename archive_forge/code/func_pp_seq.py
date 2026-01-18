import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_seq(self, a, d, xs):
    return self.pp_seq_core(self.pp_expr, a, d, xs)