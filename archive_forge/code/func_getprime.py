from rsa._compat import range
import rsa.common
import rsa.randnum
def getprime(nbits):
    """Returns a prime number that can be stored in 'nbits' bits.

    >>> p = getprime(128)
    >>> is_prime(p-1)
    False
    >>> is_prime(p)
    True
    >>> is_prime(p+1)
    False

    >>> from rsa import common
    >>> common.bit_size(p) == 128
    True
    """
    assert nbits > 3
    while True:
        integer = rsa.randnum.read_random_odd_int(nbits)
        if is_prime(integer):
            return integer