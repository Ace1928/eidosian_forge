from __future__ import annotations
from sympy.core.function import Function
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity, perfect_power
from sympy.utilities.misc import as_int
from sympy.core.random import _randint, randint
from itertools import cycle, product
def n_order(a, n):
    """Returns the order of ``a`` modulo ``n``.

    The order of ``a`` modulo ``n`` is the smallest integer
    ``k`` such that ``a**k`` leaves a remainder of 1 with ``n``.

    Parameters
    ==========

    a : integer
    n : integer, n > 1. a and n should be relatively prime

    Examples
    ========

    >>> from sympy.ntheory import n_order
    >>> n_order(3, 7)
    6
    >>> n_order(4, 7)
    3
    """
    from collections import defaultdict
    a, n = (as_int(a), as_int(n))
    if n <= 1:
        raise ValueError('n should be an integer greater than 1')
    a = a % n
    if a == 1:
        return 1
    if igcd(a, n) != 1:
        raise ValueError('The two numbers should be relatively prime')
    factors = defaultdict(int)
    for px, kx in factorint(n).items():
        if kx > 1:
            factors[px] += kx - 1
        for py, ky in factorint(px - 1).items():
            factors[py] += ky
    order = 1
    for px, kx in factors.items():
        order *= px ** kx
    for p, e in factors.items():
        for _ in range(e):
            if pow(a, order // p, n) == 1:
                order //= p
            else:
                break
    return order