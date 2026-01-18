import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_quantifier(self, a, d, xs):
    ys = [to_format(a.var_name(i)) for i in range(a.num_vars())]
    new_xs = xs + ys
    body_pp = self.pp_expr(a.body(), d + 1, new_xs)
    ys_pp = group(seq(ys))
    if a.is_forall():
        header = '&forall;'
    else:
        header = '&exist;'
    return group(compose(to_format(header, 1), indent(1, compose(ys_pp, to_format(' :'), line_break(), body_pp))))