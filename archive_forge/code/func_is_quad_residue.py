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
def is_quad_residue(a, p):
    """
    Returns True if ``a`` (mod ``p``) is in the set of squares mod ``p``,
    i.e a % p in set([i**2 % p for i in range(p)]).

    Examples
    ========

    If ``p`` is an odd
    prime, an iterative method is used to make the determination:

    >>> from sympy.ntheory import is_quad_residue
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]
    >>> [j for j in range(7) if is_quad_residue(j, 7)]
    [0, 1, 2, 4]

    See Also
    ========

    legendre_symbol, jacobi_symbol
    """
    a, p = (as_int(a), as_int(p))
    if p < 1:
        raise ValueError('p must be > 0')
    if a >= p or a < 0:
        a = a % p
    if a < 2 or p < 3:
        return True
    if not isprime(p):
        if p % 2 and jacobi_symbol(a, p) == -1:
            return False
        r = sqrt_mod(a, p)
        if r is None:
            return False
        else:
            return True
    return pow(a, (p - 1) // 2, p) == 1