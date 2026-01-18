import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
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