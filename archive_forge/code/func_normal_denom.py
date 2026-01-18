from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def normal_denom(fa, fd, ga, gd, DE):
    """
    Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g in k(t) with f weakly
    normalized with respect to t, either raise NonElementaryIntegralException,
    in which case the equation Dy + f*y == g has no solution in k(t), or the
    quadruplet (a, b, c, h) such that a, h in k[t], b, c in k<t>, and for any
    solution y in k(t) of Dy + f*y == g, q = y*h in k<t> satisfies
    a*Dq + b*q == c.

    This constitutes step 1 in the outline given in the rde.py docstring.
    """
    dn, ds = splitfactor(fd, DE)
    en, es = splitfactor(gd, DE)
    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))
    a = dn * h
    c = a * h
    if c.div(en)[1]:
        raise NonElementaryIntegralException
    ca = c * ga
    ca, cd = ca.cancel(gd, include=True)
    ba = a * fa - dn * derivation(h, DE) * fd
    ba, bd = ba.cancel(fd, include=True)
    return (a, (ba, bd), (ca, cd), h)