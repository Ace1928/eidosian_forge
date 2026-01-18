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
def lfsr_sequence(key, fill, n):
    """
    This function creates an LFSR sequence.

    Parameters
    ==========

    key : list
        A list of finite field elements, `[c_0, c_1, \\ldots, c_k].`

    fill : list
        The list of the initial terms of the LFSR sequence,
        `[x_0, x_1, \\ldots, x_k].`

    n
        Number of terms of the sequence that the function returns.

    Returns
    =======

    L
        The LFSR sequence defined by
        `x_{n+1} = c_k x_n + \\ldots + c_0 x_{n-k}`, for
        `n \\leq k`.

    Notes
    =====

    S. Golomb [G]_ gives a list of three statistical properties a
    sequence of numbers `a = \\{a_n\\}_{n=1}^\\infty`,
    `a_n \\in \\{0,1\\}`, should display to be considered
    "random". Define the autocorrelation of `a` to be

    .. math::

        C(k) = C(k,a) = \\lim_{N\\rightarrow \\infty} {1\\over N}\\sum_{n=1}^N (-1)^{a_n + a_{n+k}}.

    In the case where `a` is periodic with period
    `P` then this reduces to

    .. math::

        C(k) = {1\\over P}\\sum_{n=1}^P (-1)^{a_n + a_{n+k}}.

    Assume `a` is periodic with period `P`.

    - balance:

      .. math::

        \\left|\\sum_{n=1}^P(-1)^{a_n}\\right| \\leq 1.

    - low autocorrelation:

       .. math::

         C(k) = \\left\\{ \\begin{array}{cc} 1,& k = 0,\\\\ \\epsilon, & k \\ne 0. \\end{array} \\right.

      (For sequences satisfying these first two properties, it is known
      that `\\epsilon = -1/P` must hold.)

    - proportional runs property: In each period, half the runs have
      length `1`, one-fourth have length `2`, etc.
      Moreover, there are as many runs of `1`'s as there are of
      `0`'s.

    Examples
    ========

    >>> from sympy.crypto.crypto import lfsr_sequence
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> lfsr_sequence(key, fill, 10)
    [1 mod 2, 1 mod 2, 0 mod 2, 1 mod 2, 0 mod 2,
    1 mod 2, 1 mod 2, 0 mod 2, 0 mod 2, 1 mod 2]

    References
    ==========

    .. [G] Solomon Golomb, Shift register sequences, Aegean Park Press,
       Laguna Hills, Ca, 1967

    """
    if not isinstance(key, list):
        raise TypeError('key must be a list')
    if not isinstance(fill, list):
        raise TypeError('fill must be a list')
    p = key[0].mod
    F = FF(p)
    s = fill
    k = len(fill)
    L = []
    for i in range(n):
        s0 = s[:]
        L.append(s[0])
        s = s[1:k]
        x = sum([int(key[i] * s0[i]) for i in range(k)])
        s.append(F(x))
    return L