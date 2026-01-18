import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_atmost(self, a, d, f, xs):
    k = Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, 0)
    return seq1(self.pp_name(a), [seq3([self.pp_expr(ch, d + 1, xs) for ch in a.children()]), to_format(k)])