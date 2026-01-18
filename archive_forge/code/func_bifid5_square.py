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
def bifid5_square(key=None):
    """
    5x5 Polybius square.

    Produce the Polybius square for the `5 \\times 5` Bifid cipher.

    Examples
    ========

    >>> from sympy.crypto.crypto import bifid5_square
    >>> bifid5_square("gold bug")
    Matrix([
    [G, O, L, D, B],
    [U, A, C, E, F],
    [H, I, K, M, N],
    [P, Q, R, S, T],
    [V, W, X, Y, Z]])

    """
    if not key:
        key = bifid5
    else:
        _, key, _ = _prep('', key.upper(), None, bifid5)
        key = padded_key(key, bifid5)
    return bifid_square(key)