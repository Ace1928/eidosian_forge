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
def primitive_root(p):
    """
    Returns the smallest primitive root or None.

    Parameters
    ==========

    p : positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import primitive_root
    >>> primitive_root(19)
    2

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44
    .. [2] P. Hackman "Elementary Number Theory" (2009), Chapter C

    """
    p = as_int(p)
    if p < 1:
        raise ValueError('p is required to be positive')
    if p <= 2:
        return 1
    f = factorint(p)
    if len(f) > 2:
        return None
    if len(f) == 2:
        if 2 not in f or f[2] > 1:
            return None
        for p1, e1 in f.items():
            if p1 != 2:
                break
        i = 1
        while i < p:
            i += 2
            if i % p1 == 0:
                continue
            if is_primitive_root(i, p):
                return i
    else:
        if 2 in f:
            if p == 4:
                return 3
            return None
        p1, n = list(f.items())[0]
        if n > 1:
            g = primitive_root(p1)
            if is_primitive_root(g, p1 ** 2):
                return g
            else:
                for i in range(2, g + p1 + 1):
                    if igcd(i, p) == 1 and is_primitive_root(i, p):
                        return i
    return next(_primitive_root_prime_iter(p))