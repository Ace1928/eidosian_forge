import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def calculate_keys(p, q):
    """Calculates an encryption and a decryption key given p and q, and
    returns them as a tuple (e, d)

    :param p: the first large prime
    :param q: the second large prime

    :return: tuple (e, d) with the encryption and decryption exponents.
    """
    return calculate_keys_custom_exponent(p, q, DEFAULT_EXPONENT)