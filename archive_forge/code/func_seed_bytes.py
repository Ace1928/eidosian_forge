from binascii import hexlify
import hashlib
import logging; log = logging.getLogger(__name__)
import struct
import warnings
from passlib import exc
from passlib.utils import getrandbytes
from passlib.utils.compat import PYPY, u, bascii_to_str
from passlib.utils.decor import classproperty
from passlib.tests.utils import TestCase, skipUnless, TEST_MODE, hb
from passlib.crypto import scrypt as scrypt_mod
def seed_bytes(seed, count):
    """
    generate random reference bytes from specified seed.
    used to generate some predictable test vectors.
    """
    if hasattr(seed, 'encode'):
        seed = seed.encode('ascii')
    buf = b''
    i = 0
    while len(buf) < count:
        buf += hashlib.sha256(seed + struct.pack('<I', i)).digest()
        i += 1
    return buf[:count]