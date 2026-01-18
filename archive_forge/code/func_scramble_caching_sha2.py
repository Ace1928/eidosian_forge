from .err import OperationalError
from functools import partial
import hashlib
def scramble_caching_sha2(password, nonce):
    """Scramble algorithm used in cached_sha2_password fast path.

    XOR(SHA256(password), SHA256(SHA256(SHA256(password)), nonce))
    """
    if not password:
        return b''
    p1 = hashlib.sha256(password).digest()
    p2 = hashlib.sha256(p1).digest()
    p3 = hashlib.sha256(p2 + nonce).digest()
    res = bytearray(p1)
    for i in range(len(p3)):
        res[i] ^= p3[i]
    return bytes(res)