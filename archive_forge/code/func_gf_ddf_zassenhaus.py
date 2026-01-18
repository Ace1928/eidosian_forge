from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_ddf_zassenhaus(f, p, K):
    """
    Cantor-Zassenhaus: Deterministic Distinct Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes
    partial distinct degree factorization ``f_1 ... f_d`` of ``f`` where
    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a
    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``
    is an argument to the equal degree factorization routine.

    Consider the polynomial ``x**15 - 1`` in ``GF(11)[x]``::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_from_dict

       >>> f = gf_from_dict({15: ZZ(1), 0: ZZ(-1)}, 11, ZZ)

    Distinct degree factorization gives::

       >>> from sympy.polys.galoistools import gf_ddf_zassenhaus

       >>> gf_ddf_zassenhaus(f, 11, ZZ)
       [([1, 0, 0, 0, 0, 10], 1), ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 2)]

    which means ``x**15 - 1 = (x**5 - 1) (x**10 + x**5 + 1)``. To obtain
    factorization into irreducibles, use equal degree factorization
    procedure (EDF) with each of the factors.

    References
    ==========

    .. [1] [Gathen99]_
    .. [2] [Geddes92]_

    """
    i, g, factors = (1, [K.one, K.zero], [])
    b = gf_frobenius_monomial_base(f, p, K)
    while 2 * i <= gf_degree(f):
        g = gf_frobenius_map(g, f, b, p, K)
        h = gf_gcd(f, gf_sub(g, [K.one, K.zero], p, K), p, K)
        if h != [K.one]:
            factors.append((h, i))
            f = gf_quo(f, h, p, K)
            g = gf_rem(g, f, p, K)
            b = gf_frobenius_monomial_base(f, p, K)
        i += 1
    if f != [K.one]:
        return factors + [(f, gf_degree(f))]
    else:
        return factors