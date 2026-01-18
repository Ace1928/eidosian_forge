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
def gruntz(e, z, z0, dir='+'):
    """
    Compute the limit of e(z) at the point z0 using the Gruntz algorithm.

    Explanation
    ===========

    ``z0`` can be any expression, including oo and -oo.

    For ``dir="+"`` (default) it calculates the limit from the right
    (z->z0+) and for ``dir="-"`` the limit from the left (z->z0-). For infinite z0
    (oo or -oo), the dir argument does not matter.

    This algorithm is fully described in the module docstring in the gruntz.py
    file. It relies heavily on the series expansion. Most frequently, gruntz()
    is only used if the faster limit() function (which uses heuristics) fails.
    """
    if not z.is_symbol:
        raise NotImplementedError('Second argument must be a Symbol')
    r = None
    if z0 in (oo, I * oo):
        e0 = e
    elif z0 in (-oo, -I * oo):
        e0 = e.subs(z, -z)
    elif str(dir) == '-':
        e0 = e.subs(z, z0 - 1 / z)
    elif str(dir) == '+':
        e0 = e.subs(z, z0 + 1 / z)
    else:
        raise NotImplementedError("dir must be '+' or '-'")
    r = limitinf(e0, z)
    return r.rewrite('intractable', deep=True)