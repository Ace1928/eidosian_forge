import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_loop(self, a, d, xs):
    low = Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, 0)
    arg = self.pp_expr(a.arg(0), d + 1, xs)
    if Z3_get_decl_num_parameters(a.ctx_ref(), a.decl().ast) > 1:
        high = Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, 1)
        return seq1('Loop', [arg, to_format(low), to_format(high)])
    return seq1('Loop', [arg, to_format(low)])