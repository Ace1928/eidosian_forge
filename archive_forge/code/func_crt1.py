from functools import reduce
from math import prod
from sympy.core.numbers import igcdex, igcd
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
from sympy.utilities.misc import as_int
def crt1(m):
    """First part of Chinese Remainder Theorem, for multiple application.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt1
    >>> crt1([18, 42, 6])
    (4536, [252, 108, 756], [0, 2, 0])
    """
    return gf_crt1(m, ZZ)