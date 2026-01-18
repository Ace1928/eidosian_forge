from sympy.core import S, Add, sympify, Function, Lambda, Dummy
from sympy.core.traversal import preorder_traversal
from sympy.polys import Poly, RootSum, cancel, factor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyoptions import allowed_flags, set_defaults
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.utilities import numbered_symbols, take, xthreaded, public
def apart_list_full_decomposition(P, Q, dummygen):
    """
    Bronstein's full partial fraction decomposition algorithm.

    Given a univariate rational function ``f``, performing only GCD
    operations over the algebraic closure of the initial ground domain
    of definition, compute full partial fraction decomposition with
    fractions having linear denominators.

    Note that no factorization of the initial denominator of ``f`` is
    performed. The final decomposition is formed in terms of a sum of
    :class:`RootSum` instances.

    References
    ==========

    .. [1] [Bronstein93]_

    """
    f, x, U = (P / Q, P.gen, [])
    u = Function('u')(x)
    a = Dummy('a')
    partial = []
    for d, n in Q.sqf_list_include(all=True):
        b = d.as_expr()
        U += [u.diff(x, n - 1)]
        h = cancel(f * b ** n) / u ** n
        H, subs = ([h], [])
        for j in range(1, n):
            H += [H[-1].diff(x) / j]
        for j in range(1, n + 1):
            subs += [(U[j - 1], b.diff(x, j) / j)]
        for j in range(0, n):
            P, Q = cancel(H[j]).as_numer_denom()
            for i in range(0, j + 1):
                P = P.subs(*subs[j - i])
            Q = Q.subs(*subs[0])
            P = Poly(P, x)
            Q = Poly(Q, x)
            G = P.gcd(d)
            D = d.quo(G)
            B, g = Q.half_gcdex(D)
            b = (P * B.quo(g)).rem(D)
            Dw = D.subs(x, next(dummygen))
            numer = Lambda(a, b.as_expr().subs(x, a))
            denom = Lambda(a, x - a)
            exponent = n - j
            partial.append((Dw, numer, denom, exponent))
    return partial