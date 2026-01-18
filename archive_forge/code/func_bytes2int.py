from __future__ import absolute_import
import binascii
from struct import pack
from rsa._compat import byte, is_integer
from rsa import common, machine_size
def bytes2int(raw_bytes):
    """Converts a list of bytes or an 8-bit string to an integer.

    When using unicode strings, encode it to some encoding like UTF8 first.

    >>> (((128 * 256) + 64) * 256) + 15
    8405007
    >>> bytes2int(b'\\x80@\\x0f')
    8405007

    """
    return int(binascii.hexlify(raw_bytes), 16)