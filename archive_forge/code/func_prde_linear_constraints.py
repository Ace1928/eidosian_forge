import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def prde_linear_constraints(a, b, G, DE):
    """
    Parametric Risch Differential Equation - Generate linear constraints on the constants.

    Explanation
    ===========

    Given a derivation D on k[t], a, b, in k[t] with gcd(a, b) == 1, and
    G = [g1, ..., gm] in k(t)^m, return Q = [q1, ..., qm] in k[t]^m and a
    matrix M with entries in k(t) such that for any solution c1, ..., cm in
    Const(k) and p in k[t] of a*Dp + b*p == Sum(ci*gi, (i, 1, m)),
    (c1, ..., cm) is a solution of Mx == 0, and p and the ci satisfy
    a*Dp + b*p == Sum(ci*qi, (i, 1, m)).

    Because M has entries in k(t), and because Matrix does not play well with
    Poly, M will be a Matrix of Basic expressions.
    """
    m = len(G)
    Gns, Gds = list(zip(*G))
    d = reduce(lambda i, j: i.lcm(j), Gds)
    d = Poly(d, field=True)
    Q = [(ga * d.quo(gd)).div(d) for ga, gd in G]
    if not all((ri.is_zero for _, ri in Q)):
        N = max((ri.degree(DE.t) for _, ri in Q))
        M = Matrix(N + 1, m, lambda i, j: Q[j][1].nth(i), DE.t)
    else:
        M = Matrix(0, m, [], DE.t)
    qs, _ = list(zip(*Q))
    return (qs, M)