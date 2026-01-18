import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def prde_normal_denom(fa, fd, G, DE):
    """
    Parametric Risch Differential Equation - Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g1, ..., gm in k(t) with f weakly
    normalized with respect to t, return the tuple (a, b, G, h) such that
    a, h in k[t], b in k<t>, G = [g1, ..., gm] in k(t)^m, and for any solution
    c1, ..., cm in Const(k) and y in k(t) of Dy + f*y == Sum(ci*gi, (i, 1, m)),
    q == y*h in k<t> satisfies a*Dq + b*q == Sum(ci*Gi, (i, 1, m)).
    """
    dn, ds = splitfactor(fd, DE)
    Gas, Gds = list(zip(*G))
    gd = reduce(lambda i, j: i.lcm(j), Gds, Poly(1, DE.t))
    en, es = splitfactor(gd, DE)
    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))
    a = dn * h
    c = a * h
    ba = a * fa - dn * derivation(h, DE) * fd
    ba, bd = ba.cancel(fd, include=True)
    G = [(c * A).cancel(D, include=True) for A, D in G]
    return (a, (ba, bd), G, h)