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
def mrv_max1(f, g, exps, x):
    """Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. mrv_max1() compares (two elements of)
    f and g and returns the set, which is in the higher comparability class
    of the union of both, if they have the same order of variation.
    Also returns exps, with the appropriate substitutions made.
    """
    u, b = f.union(g, exps)
    return mrv_max3(f, g.do_subs(exps), g, f.do_subs(exps), u, b, x)