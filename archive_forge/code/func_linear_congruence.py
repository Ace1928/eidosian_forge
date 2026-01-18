from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def linear_congruence(a, b, m):
    """
    Returns the values of x satisfying a*x congruent b mod(m)

    Here m is positive integer and a, b are natural numbers.
    This function returns only those values of x which are distinct mod(m).

    Examples
    ========

    >>> from sympy.polys.galoistools import linear_congruence

    >>> linear_congruence(3, 12, 15)
    [4, 9, 14]

    There are 3 solutions distinct mod(15) since gcd(a, m) = gcd(3, 15) = 3.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Linear_congruence_theorem

    """
    from sympy.polys.polytools import gcdex
    if a % m == 0:
        if b % m == 0:
            return list(range(m))
        else:
            return []
    r, _, g = gcdex(a, m)
    if b % g != 0:
        return []
    return [(r * b // g + t * m // g) % m for t in range(g)]