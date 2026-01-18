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
def bifid6_square(key=None):
    """
    6x6 Polybius square.

    Produces the Polybius square for the `6 \\times 6` Bifid cipher.
    Assumes alphabet of symbols is "A", ..., "Z", "0", ..., "9".

    Examples
    ========

    >>> from sympy.crypto.crypto import bifid6_square
    >>> key = "gold bug"
    >>> bifid6_square(key)
    Matrix([
    [G, O, L, D, B, U],
    [A, C, E, F, H, I],
    [J, K, M, N, P, Q],
    [R, S, T, V, W, X],
    [Y, Z, 0, 1, 2, 3],
    [4, 5, 6, 7, 8, 9]])

    """
    if not key:
        key = bifid6
    else:
        _, key, _ = _prep('', key.upper(), None, bifid6)
        key = padded_key(key, bifid6)
    return bifid_square(key)