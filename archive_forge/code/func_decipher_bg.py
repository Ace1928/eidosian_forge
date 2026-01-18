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
def decipher_bg(message, key):
    """
    Decrypts the message using private keys.

    Explanation
    ===========

    ALGORITHM:
        1. Let, c be the encrypted message, y the second number received,
        and p and q be the private keys.
        2. Compute, r_p = y^((p+1)/4 ^ L) mod p and
        r_q = y^((q+1)/4 ^ L) mod q.
        3. Compute x_0 = (q(q^-1 mod p)r_p + p(p^-1 mod q)r_q) mod N.
        4. From, recompute the bits using the BBS generator, as in the
        encryption algorithm.
        5. Compute original message by XORing c and b.

    Parameters
    ==========

    message
        Tuple of encrypted message and a non-negative integer.

    key
        Tuple of private keys.

    Returns
    =======

    orig_msg
        The original message

    """
    p, q = key
    encrypt_msg, y = message
    public_key = p * q
    L = len(encrypt_msg)
    p_t = ((p + 1) / 4) ** L
    q_t = ((q + 1) / 4) ** L
    r_p = pow(int(y), int(p_t), int(p))
    r_q = pow(int(y), int(q_t), int(q))
    x = (q * mod_inverse(q, p) * r_p + p * mod_inverse(p, q) * r_q) % public_key
    orig_bits = []
    for _ in range(L):
        orig_bits.append(x % 2)
        x = x ** 2 % public_key
    orig_msg = 0
    for m, b in zip(encrypt_msg, orig_bits):
        orig_msg = orig_msg * 2
        orig_msg += m ^ b
    return orig_msg