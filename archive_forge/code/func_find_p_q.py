import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def find_p_q(nbits, getprime_func=rsa.prime.getprime, accurate=True):
    """Returns a tuple of two different primes of nbits bits each.

    The resulting p * q has exacty 2 * nbits bits, and the returned p and q
    will not be equal.

    :param nbits: the number of bits in each of p and q.
    :param getprime_func: the getprime function, defaults to
        :py:func:`rsa.prime.getprime`.

        *Introduced in Python-RSA 3.1*

    :param accurate: whether to enable accurate mode or not.
    :returns: (p, q), where p > q

    >>> (p, q) = find_p_q(128)
    >>> from rsa import common
    >>> common.bit_size(p * q)
    256

    When not in accurate mode, the number of bits can be slightly less

    >>> (p, q) = find_p_q(128, accurate=False)
    >>> from rsa import common
    >>> common.bit_size(p * q) <= 256
    True
    >>> common.bit_size(p * q) > 240
    True

    """
    total_bits = nbits * 2
    shift = nbits // 16
    pbits = nbits + shift
    qbits = nbits - shift
    log.debug('find_p_q(%i): Finding p', nbits)
    p = getprime_func(pbits)
    log.debug('find_p_q(%i): Finding q', nbits)
    q = getprime_func(qbits)

    def is_acceptable(p, q):
        """Returns True iff p and q are acceptable:

            - p and q differ
            - (p * q) has the right nr of bits (when accurate=True)
        """
        if p == q:
            return False
        if not accurate:
            return True
        found_size = rsa.common.bit_size(p * q)
        return total_bits == found_size
    change_p = False
    while not is_acceptable(p, q):
        if change_p:
            p = getprime_func(pbits)
        else:
            q = getprime_func(qbits)
        change_p = not change_p
    return (max(p, q), min(p, q))