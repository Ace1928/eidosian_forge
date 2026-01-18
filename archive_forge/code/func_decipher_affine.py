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
def decipher_affine(msg, key, symbols=None):
    """
    Return the deciphered text that was made from the mapping,
    `x \\rightarrow ax+b` (mod `N`), where ``N`` is the
    number of characters in the alphabet. Deciphering is done by
    reciphering with a new key: `x \\rightarrow cx+d` (mod `N`),
    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_affine, decipher_affine
    >>> msg = "GO NAVY BEAT ARMY"
    >>> key = (3, 1)
    >>> encipher_affine(msg, key)
    'TROBMVENBGBALV'
    >>> decipher_affine(_, key)
    'GONAVYBEATARMY'

    See Also
    ========

    encipher_affine

    """
    return encipher_affine(msg, key, symbols, _inverse=True)