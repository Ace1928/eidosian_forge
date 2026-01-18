from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings
from itertools import cycle
from sympy.core import Symbol
from sympy.core.numbers import igcdex, mod_inverse, igcd, Rational
from sympy.core.random import _randrange, _randint
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory import totient as _euler
from sympy.ntheory import reduced_totient as _carmichael
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import gcd, Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
def lfsr_connection_polynomial(s):
    """
    This function computes the LFSR connection polynomial.

    Parameters
    ==========

    s
        A sequence of elements of even length, with entries in a finite
        field.

    Returns
    =======

    C(x)
        The connection polynomial of a minimal LFSR yielding s.

        This implements the algorithm in section 3 of J. L. Massey's
        article [M]_.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     lfsr_sequence, lfsr_connection_polynomial)
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**4 + x + 1
    >>> fill = [F(1), F(0), F(0), F(1)]
    >>> key = [F(1), F(1), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + 1
    >>> fill = [F(1), F(0), F(1)]
    >>> key = [F(1), F(1), F(0)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + x**2 + 1
    >>> fill = [F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + x + 1

    References
    ==========

    .. [M] James L. Massey, "Shift-Register Synthesis and BCH Decoding."
        IEEE Trans. on Information Theory, vol. 15(1), pp. 122-127,
        Jan 1969.

    """
    p = s[0].mod
    x = Symbol('x')
    C = 1 * x ** 0
    B = 1 * x ** 0
    m = 1
    b = 1 * x ** 0
    L = 0
    N = 0
    while N < len(s):
        if L > 0:
            dC = Poly(C).degree()
            r = min(L + 1, dC + 1)
            coeffsC = [C.subs(x, 0)] + [C.coeff(x ** i) for i in range(1, dC + 1)]
            d = (s[N].to_int() + sum([coeffsC[i] * s[N - i].to_int() for i in range(1, r)])) % p
        if L == 0:
            d = s[N].to_int() * x ** 0
        if d == 0:
            m += 1
            N += 1
        if d > 0:
            if 2 * L > N:
                C = (C - d * (b ** (p - 2) % p) * x ** m * B).expand()
                m += 1
                N += 1
            else:
                T = C
                C = (C - d * (b ** (p - 2) % p) * x ** m * B).expand()
                L = N + 1 - L
                m = 1
                b = d
                B = T
                N += 1
    dC = Poly(C).degree()
    coeffsC = [C.subs(x, 0)] + [C.coeff(x ** i) for i in range(1, dC + 1)]
    return sum([coeffsC[i] % p * x ** i for i in range(dC + 1) if coeffsC[i] is not None])