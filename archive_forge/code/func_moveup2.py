from functools import reduce
from sympy.core import Basic, S, Mul, PoleError, expand_mul
from sympy.core.cache import cacheit
from sympy.core.numbers import ilcm, I, oo
from sympy.core.symbol import Dummy, Wild
from sympy.core.traversal import bottom_up
from sympy.functions import log, exp, sign as _sign
from sympy.series.order import Order
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.misc import debug_decorator as debug
from sympy.utilities.timeutils import timethis
def moveup2(s, x):
    r = SubsSet()
    for expr, var in s.items():
        r[expr.xreplace({x: exp(x)})] = var
    for var, expr in s.rewrites.items():
        r.rewrites[var] = s.rewrites[var].xreplace({x: exp(x)})
    return r