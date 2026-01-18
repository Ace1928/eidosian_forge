import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def gen_keys(nbits, getprime_func, accurate=True, exponent=DEFAULT_EXPONENT):
    """Generate RSA keys of nbits bits. Returns (p, q, e, d).

    Note: this can take a long time, depending on the key size.

    :param nbits: the total number of bits in ``p`` and ``q``. Both ``p`` and
        ``q`` will use ``nbits/2`` bits.
    :param getprime_func: either :py:func:`rsa.prime.getprime` or a function
        with similar signature.
    :param exponent: the exponent for the key; only change this if you know
        what you're doing, as the exponent influences how difficult your
        private key can be cracked. A very common choice for e is 65537.
    :type exponent: int
    """
    while True:
        p, q = find_p_q(nbits // 2, getprime_func, accurate)
        try:
            e, d = calculate_keys_custom_exponent(p, q, exponent=exponent)
            break
        except ValueError:
            pass
    return (p, q, e, d)