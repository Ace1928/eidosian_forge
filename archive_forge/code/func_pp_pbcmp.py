import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_pbcmp(self, a, d, f, xs):
    chs = a.children()
    rchs = range(len(chs))
    k = Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, 0)
    ks = [Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, i + 1) for i in rchs]
    ls = [seq3([self.pp_expr(chs[i], d + 1, xs), to_format(ks[i])]) for i in rchs]
    return seq1(self.pp_name(a), [seq3(ls), to_format(k)])