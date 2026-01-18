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
def encipher_bifid6(msg, key):
    """
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    This is the version of the Bifid cipher that uses the `6 \\times 6`
    Polybius square.

    Parameters
    ==========

    msg
        Plaintext string (digits okay).

    key
        Short string for key (digits okay).

        If ``key`` is less than 36 characters long, the square will be
        filled with letters A through Z and digits 0 through 9.

    Returns
    =======

    ciphertext
        Ciphertext from Bifid cipher (all caps, no spaces).

    See Also
    ========

    decipher_bifid6, encipher_bifid

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    key = padded_key(key, bifid6)
    return encipher_bifid(msg, '', key)