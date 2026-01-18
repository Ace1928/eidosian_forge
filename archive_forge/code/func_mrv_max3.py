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
def mrv_max3(f, expsf, g, expsg, union, expsboth, x):
    """
    Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. max() compares (two elements of)
    f and g and returns either (f, expsf) [if f is larger], (g, expsg)
    [if g is larger] or (union, expsboth) [if f, g are of the same class].
    """
    if not isinstance(f, SubsSet):
        raise TypeError('f should be an instance of SubsSet')
    if not isinstance(g, SubsSet):
        raise TypeError('g should be an instance of SubsSet')
    if f == SubsSet():
        return (g, expsg)
    elif g == SubsSet():
        return (f, expsf)
    elif f.meets(g):
        return (union, expsboth)
    c = compare(list(f.keys())[0], list(g.keys())[0], x)
    if c == '>':
        return (f, expsf)
    elif c == '<':
        return (g, expsg)
    else:
        if c != '=':
            raise ValueError('c should be =')
        return (union, expsboth)