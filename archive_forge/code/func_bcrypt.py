import re
import struct
from functools import reduce
from Cryptodome.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
from Cryptodome.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Cryptodome.Util.strxor import strxor
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import size as bit_size, long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def bcrypt(password, cost, salt=None):
    """Hash a password into a key, using the OpenBSD bcrypt protocol.

    Args:
      password (byte string or string):
        The secret password or pass phrase.
        It must be at most 72 bytes long.
        It must not contain the zero byte.
        Unicode strings will be encoded as UTF-8.
      cost (integer):
        The exponential factor that makes it slower to compute the hash.
        It must be in the range 4 to 31.
        A value of at least 12 is recommended.
      salt (byte string):
        Optional. Random byte string to thwarts dictionary and rainbow table
        attacks. It must be 16 bytes long.
        If not passed, a random value is generated.

    Return (byte string):
        The bcrypt hash

    Raises:
        ValueError: if password is longer than 72 bytes or if it contains the zero byte

   """
    password = tobytes(password, 'utf-8')
    if password.find(bchr(0)[0]) != -1:
        raise ValueError('The password contains the zero byte')
    if len(password) < 72:
        password += b'\x00'
    if salt is None:
        salt = get_random_bytes(16)
    if len(salt) != 16:
        raise ValueError('bcrypt salt must be 16 bytes long')
    ctext = _bcrypt_hash(password, cost, salt, b'OrpheanBeholderScryDoubt', True)
    cost_enc = b'$' + bstr(str(cost).zfill(2))
    salt_enc = b'$' + _bcrypt_encode(salt)
    hash_enc = _bcrypt_encode(ctext[:-1])
    return b'$2a' + cost_enc + salt_enc + hash_enc