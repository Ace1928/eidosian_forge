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
def encipher_elgamal(i, key, seed=None):
    """
    Encrypt message with public key.

    Explanation
    ===========

    ``i`` is a plaintext message expressed as an integer.
    ``key`` is public key (p, r, e). In order to encrypt
    a message, a random number ``a`` in ``range(2, p)``
    is generated and the encryped message is returned as
    `c_{1}` and `c_{2}` where:

    `c_{1} \\equiv r^{a} \\pmod p`

    `c_{2} \\equiv m e^{a} \\pmod p`

    Parameters
    ==========

    msg
        int of encoded message.

    key
        Public key.

    Returns
    =======

    tuple : (c1, c2)
        Encipher into two number.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_elgamal, elgamal_private_key, elgamal_public_key
    >>> pri = elgamal_private_key(5, seed=[3]); pri
    (37, 2, 3)
    >>> pub = elgamal_public_key(pri); pub
    (37, 2, 8)
    >>> msg = 36
    >>> encipher_elgamal(msg, pub, seed=[3])
    (8, 6)

    """
    p, r, e = key
    if i < 0 or i >= p:
        raise ValueError('Message (%s) should be in range(%s)' % (i, p))
    randrange = _randrange(seed)
    a = randrange(2, p)
    return (pow(r, a, p), i * pow(e, a, p) % p)