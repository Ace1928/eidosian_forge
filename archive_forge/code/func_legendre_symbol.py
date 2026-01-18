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
def legendre_symbol(a, p):
    """
    Returns the Legendre symbol `(a / p)`.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \\genfrac(){}{}{a}{p} = \\begin{cases}
             0 & \\text{if } p \\text{ divides } a\\\\
             1 & \\text{if } a \\text{ is a quadratic residue modulo } p\\\\
            -1 & \\text{if } a \\text{ is a quadratic nonresidue modulo } p
        \\end{cases}

    Parameters
    ==========

    a : integer
    p : odd prime

    Examples
    ========

    >>> from sympy.ntheory import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    is_quad_residue, jacobi_symbol

    """
    a, p = (as_int(a), as_int(p))
    if not isprime(p) or p == 2:
        raise ValueError('p should be an odd prime')
    a = a % p
    if not a:
        return 0
    if pow(a, (p - 1) // 2, p) == 1:
        return 1
    return -1