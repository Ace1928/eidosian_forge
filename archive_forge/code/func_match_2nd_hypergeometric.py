from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants
def match_2nd_hypergeometric(eq, func):
    x = func.args[0]
    df = func.diff(x)
    a3 = Wild('a3', exclude=[func, func.diff(x), func.diff(x, 2)])
    b3 = Wild('b3', exclude=[func, func.diff(x), func.diff(x, 2)])
    c3 = Wild('c3', exclude=[func, func.diff(x), func.diff(x, 2)])
    deq = a3 * func.diff(x, 2) + b3 * df + c3 * func
    r = collect(eq, [func.diff(x, 2), func.diff(x), func]).match(deq)
    if r:
        if not all((val.is_polynomial() for val in r.values())):
            n, d = eq.as_numer_denom()
            eq = expand(n)
            r = collect(eq, [func.diff(x, 2), func.diff(x), func]).match(deq)
    if r and r[a3] != 0:
        A = cancel(r[b3] / r[a3])
        B = cancel(r[c3] / r[a3])
        return [A, B]
    else:
        return []