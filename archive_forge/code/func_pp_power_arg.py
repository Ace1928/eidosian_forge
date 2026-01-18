import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_power_arg(self, arg, d, xs):
    r = self.pp_expr(arg, d + 1, xs)
    k = None
    if z3.is_app(arg):
        k = arg.decl().kind()
    if self.is_infix_unary(k) or (z3.is_rational_value(arg) and arg.denominator_as_long() != 1):
        return self.add_paren(r)
    else:
        return r