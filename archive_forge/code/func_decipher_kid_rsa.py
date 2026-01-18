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
def decipher_kid_rsa(msg, key):
    """
    Here ``msg`` is the plaintext and ``key`` is the private key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     kid_rsa_public_key, kid_rsa_private_key,
    ...     decipher_kid_rsa, encipher_kid_rsa)
    >>> a, b, A, B = 3, 4, 5, 6
    >>> d = kid_rsa_private_key(a, b, A, B)
    >>> msg = 200
    >>> pub = kid_rsa_public_key(a, b, A, B)
    >>> pri = kid_rsa_private_key(a, b, A, B)
    >>> ct = encipher_kid_rsa(msg, pub)
    >>> decipher_kid_rsa(ct, pri)
    200

    """
    n, d = key
    return msg * d % n