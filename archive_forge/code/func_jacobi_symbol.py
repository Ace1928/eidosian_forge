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
def jacobi_symbol(m, n):
    """
    Returns the Jacobi symbol `(m / n)`.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \\genfrac(){}{}{m}{n} =
            \\genfrac(){}{}{m}{p^{1}}^{\\alpha_1}
            \\genfrac(){}{}{m}{p^{2}}^{\\alpha_2}
            ...
            \\genfrac(){}{}{m}{p^{k}}^{\\alpha_k}
            \\text{ where } n =
                p_1^{\\alpha_1}
                p_2^{\\alpha_2}
                ...
                p_k^{\\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Parameters
    ==========

    m : integer
    n : odd positive integer

    Examples
    ========

    >>> from sympy.ntheory import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    is_quad_residue, legendre_symbol
    """
    m, n = (as_int(m), as_int(n))
    if n < 0 or not n % 2:
        raise ValueError('n should be an odd positive integer')
    if m < 0 or m > n:
        m %= n
    if not m:
        return int(n == 1)
    if n == 1 or m == 1:
        return 1
    if igcd(m, n) != 1:
        return 0
    j = 1
    while m != 0:
        while m % 2 == 0 and m > 0:
            m >>= 1
            if n % 8 in [3, 5]:
                j = -j
        m, n = (n, m)
        if m % 4 == n % 4 == 3:
            j = -j
        m %= n
    return j