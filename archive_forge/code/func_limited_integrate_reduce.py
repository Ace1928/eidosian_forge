import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def limited_integrate_reduce(fa, fd, G, DE):
    """
    Simpler version of step 1 & 2 for the limited integration problem.

    Explanation
    ===========

    Given a derivation D on k(t) and f, g1, ..., gn in k(t), return
    (a, b, h, N, g, V) such that a, b, h in k[t], N is a non-negative integer,
    g in k(t), V == [v1, ..., vm] in k(t)^m, and for any solution v in k(t),
    c1, ..., cm in C of f == Dv + Sum(ci*wi, (i, 1, m)), p = v*h is in k<t>, and
    p and the ci satisfy a*Dp + b*p == g + Sum(ci*vi, (i, 1, m)).  Furthermore,
    if S1irr == Sirr, then p is in k[t], and if t is nonlinear or Liouvillian
    over k, then deg(p) <= N.

    So that the special part is always computed, this function calls the more
    general prde_special_denom() automatically if it cannot determine that
    S1irr == Sirr.  Furthermore, it will automatically call bound_degree() when
    t is linear and non-Liouvillian, which for the transcendental case, implies
    that Dt == a*t + b with for some a, b in k*.
    """
    dn, ds = splitfactor(fd, DE)
    E = [splitfactor(gd, DE) for _, gd in G]
    En, Es = list(zip(*E))
    c = reduce(lambda i, j: i.lcm(j), (dn,) + En)
    hn = c.gcd(c.diff(DE.t))
    a = hn
    b = -derivation(hn, DE)
    N = 0
    if DE.case in ('base', 'primitive', 'exp', 'tan'):
        hs = reduce(lambda i, j: i.lcm(j), (ds,) + Es)
        a = hn * hs
        b -= (hn * derivation(hs, DE)).quo(hs)
        mu = min(order_at_oo(fa, fd, DE.t), min([order_at_oo(ga, gd, DE.t) for ga, gd in G]))
        N = hn.degree(DE.t) + hs.degree(DE.t) + max(0, 1 - DE.d.degree(DE.t) - mu)
    else:
        raise NotImplementedError
    V = [(-a * hn * ga).cancel(gd, include=True) for ga, gd in G]
    return (a, b, a, N, (a * hn * fa).cancel(fd, include=True), V)