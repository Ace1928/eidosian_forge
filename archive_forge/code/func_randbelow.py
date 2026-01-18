import base64
import binascii
from hmac import compare_digest
from random import SystemRandom
def randbelow(exclusive_upper_bound):
    """Return a random int in the range [0, n)."""
    if exclusive_upper_bound <= 0:
        raise ValueError('Upper bound must be positive.')
    return _sysrand._randbelow(exclusive_upper_bound)