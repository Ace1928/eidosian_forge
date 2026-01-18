from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_factor(f, p, K):
    """
    Factor (non square-free) polynomials in ``GF(p)[x]``.

    Given a possibly non square-free polynomial ``f`` in ``GF(p)[x]``,
    returns its complete factorization into irreducibles::

                 f_1(x)**e_1 f_2(x)**e_2 ... f_d(x)**e_d

    where each ``f_i`` is a monic polynomial and ``gcd(f_i, f_j) == 1``,
    for ``i != j``.  The result is given as a tuple consisting of the
    leading coefficient of ``f`` and a list of factors of ``f`` with
    their multiplicities.

    The algorithm proceeds by first computing square-free decomposition
    of ``f`` and then iteratively factoring each of square-free factors.

    Consider a non square-free polynomial ``f = (7*x + 1) (x + 2)**2`` in
    ``GF(11)[x]``. We obtain its factorization into irreducibles as follows::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_factor

       >>> gf_factor(ZZ.map([5, 2, 7, 2]), 11, ZZ)
       (5, [([1, 2], 1), ([1, 8], 2)])

    We arrived with factorization ``f = 5 (x + 2) (x + 8)**2``. We did not
    recover the exact form of the input polynomial because we requested to
    get monic factors of ``f`` and its leading coefficient separately.

    Square-free factors of ``f`` can be factored into irreducibles over
    ``GF(p)`` using three very different methods:

    Berlekamp
        efficient for very small values of ``p`` (usually ``p < 25``)
    Cantor-Zassenhaus
        efficient on average input and with "typical" ``p``
    Shoup-Kaltofen-Gathen
        efficient with very large inputs and modulus

    If you want to use a specific factorization method, instead of the default
    one, set ``GF_FACTOR_METHOD`` with one of ``berlekamp``, ``zassenhaus`` or
    ``shoup`` values.

    References
    ==========

    .. [1] [Gathen99]_

    """
    lc, f = gf_monic(f, p, K)
    if gf_degree(f) < 1:
        return (lc, [])
    factors = []
    for g, n in gf_sqf_list(f, p, K)[1]:
        for h in gf_factor_sqf(g, p, K)[1]:
            factors.append((h, n))
    return (lc, _sort_factors(factors))