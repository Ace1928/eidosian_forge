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
def encipher_atbash(msg, symbols=None):
    """
    Enciphers a given ``msg`` into its Atbash ciphertext and returns it.

    Explanation
    ===========

    Atbash is a substitution cipher originally used to encrypt the Hebrew
    alphabet. Atbash works on the principle of mapping each alphabet to its
    reverse / counterpart (i.e. a would map to z, b to y etc.)

    Atbash is functionally equivalent to the affine cipher with ``a = 25``
    and ``b = 25``

    See Also
    ========

    decipher_atbash

    """
    return encipher_affine(msg, (25, 25), symbols)