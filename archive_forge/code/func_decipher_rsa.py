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
def decipher_rsa(i, key, factors=None):
    """Decrypt the ciphertext with RSA.

    Parameters
    ==========

    i : integer
        The ciphertext to be decrypted for.

    key : (n, d) where n, d are integers
        `n` is the modulus of the key and `d` is the exponent of the
        key. The decryption is computed by `i^d \\bmod n`.

        The key can either be a public key or a private key, however,
        the message encrypted by a public key can only be decrypted by
        a private key, and vice versa, as RSA is an asymmetric
        cryptography system.

    factors : list of coprime integers
        As the modulus `n` created from RSA key generation is composed
        of arbitrary prime factors
        `n = {p_1}^{k_1}{p_2}^{k_2}\\dots{p_n}^{k_n}` where
        `p_1, p_2, \\dots, p_n` are distinct primes and
        `k_1, k_2, \\dots, k_n` are positive integers, chinese remainder
        theorem can be used to compute `i^d \\bmod n` from the
        fragmented modulo operations like

        .. math::
            i^d \\bmod {p_1}^{k_1}, i^d \\bmod {p_2}^{k_2}, \\dots,
            i^d \\bmod {p_n}^{k_n}

        or like

        .. math::
            i^d \\bmod {p_1}^{k_1}{p_2}^{k_2},
            i^d \\bmod {p_3}^{k_3}, \\dots ,
            i^d \\bmod {p_n}^{k_n}

        as long as every moduli does not share any common divisor each
        other.

        The raw primes used in generating the RSA key pair can be a good
        option.

        Note that the speed advantage of using this is only viable for
        very large cases (Like 2048-bit RSA keys) since the
        overhead of using pure Python implementation of
        :meth:`sympy.ntheory.modular.crt` may overcompensate the
        theoretical speed advantage.

    Notes
    =====

    See the ``Notes`` section in the documentation of
    :meth:`encipher_rsa`

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_rsa, encipher_rsa
    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key

    Public Key Encryption and Decryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> new_msg = encipher_rsa(msg, prk)
    >>> new_msg
    3
    >>> decipher_rsa(new_msg, puk)
    12

    Private Key Encryption and Decryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> new_msg = encipher_rsa(msg, puk)
    >>> new_msg
    3
    >>> decipher_rsa(new_msg, prk)
    12

    Decryption using chinese remainder theorem:

    >>> decipher_rsa(new_msg, prk, factors=[p, q])
    12

    See Also
    ========

    encipher_rsa
    """
    return _encipher_decipher_rsa(i, key, factors=factors)