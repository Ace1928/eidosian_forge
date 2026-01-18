from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def encode_int64(self, value):
    """encode 64-bit integer -> 11 char hash64 string

        this format is used primarily by des-crypt & variants to encode
        the DES output value used as a checksum.
        """
    if value < 0 or value > 18446744073709551615:
        raise ValueError('value out of range')
    return self._encode_int(value, 64)