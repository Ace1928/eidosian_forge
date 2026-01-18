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
def encipher_bg(i, key, seed=None):
    """
    Encrypts the message using public key and seed.

    Explanation
    ===========

    ALGORITHM:
        1. Encodes i as a string of L bits, m.
        2. Select a random element r, where 1 < r < key, and computes
           x = r^2 mod key.
        3. Use BBS pseudo-random number generator to generate L random bits, b,
        using the initial seed as x.
        4. Encrypted message, c_i = m_i XOR b_i, 1 <= i <= L.
        5. x_L = x^(2^L) mod key.
        6. Return (c, x_L)

    Parameters
    ==========

    i
        Message, a non-negative integer

    key
        The public key

    Returns
    =======

    Tuple
        (encrypted_message, x_L)

    Raises
    ======

    ValueError
        If i is negative.

    """
    if i < 0:
        raise ValueError('message must be a non-negative integer: got %d instead' % i)
    enc_msg = []
    while i > 0:
        enc_msg.append(i % 2)
        i //= 2
    enc_msg.reverse()
    L = len(enc_msg)
    r = _randint(seed)(2, key - 1)
    x = r ** 2 % key
    x_L = pow(int(x), int(2 ** L), int(key))
    rand_bits = []
    for _ in range(L):
        rand_bits.append(x % 2)
        x = x ** 2 % key
    encrypt_msg = [m ^ b for m, b in zip(enc_msg, rand_bits)]
    return (encrypt_msg, x_L)