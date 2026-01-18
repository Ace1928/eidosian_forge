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
def decipher_gm(message, key):
    """
    Decrypt message 'message' using public_key 'key'.

    Parameters
    ==========

    message : list of int
        The randomized encrypted message.

    key : (p, q)
        The private key.

    Returns
    =======

    int
        The encrypted message.

    """
    p, q = key
    res = lambda m, p: _legendre(m, p) > 0
    bits = [res(m, p) * res(m, q) for m in message]
    m = 0
    for b in bits:
        m <<= 1
        m += not b
    return m