from __future__ import absolute_import
import binascii
from struct import pack
from rsa._compat import byte, is_integer
from rsa import common, machine_size
def bytes_leading(raw_bytes, needle=b'\x00'):
    """
    Finds the number of prefixed byte occurrences in the haystack.

    Useful when you want to deal with padding.

    :param raw_bytes:
        Raw bytes.
    :param needle:
        The byte to count. Default \x00.
    :returns:
        The number of leading needle bytes.
    """
    leading = 0
    _byte = needle[0]
    for x in raw_bytes:
        if x == _byte:
            leading += 1
        else:
            break
    return leading