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
def encipher_kid_rsa(msg, key):
    """
    Here ``msg`` is the plaintext and ``key`` is the public key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_kid_rsa, kid_rsa_public_key)
    >>> msg = 200
    >>> a, b, A, B = 3, 4, 5, 6
    >>> key = kid_rsa_public_key(a, b, A, B)
    >>> encipher_kid_rsa(msg, key)
    161

    """
    n, e = key
    return msg * e % n