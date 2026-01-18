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
def decipher_rot13(msg, symbols=None):
    """
    Performs the ROT13 decryption on a given plaintext ``msg``.

    Explanation
    ============

    ``decipher_rot13`` is equivalent to ``encipher_rot13`` as both
    ``decipher_shift`` with a key of 13 and ``encipher_shift`` key with a
    key of 13 will return the same results. Nonetheless,
    ``decipher_rot13`` has nonetheless been explicitly defined here for
    consistency.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_rot13, decipher_rot13
    >>> msg = 'GONAVYBEATARMY'
    >>> ciphertext = encipher_rot13(msg);ciphertext
    'TBANILORNGNEZL'
    >>> decipher_rot13(ciphertext)
    'GONAVYBEATARMY'
    >>> encipher_rot13(msg) == decipher_rot13(msg)
    True
    >>> msg == decipher_rot13(ciphertext)
    True

    """
    return decipher_shift(msg, 13, symbols)