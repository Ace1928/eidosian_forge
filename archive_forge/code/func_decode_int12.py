from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def decode_int12(self, source):
    """decodes 2 char string -> 12-bit integer"""
    if not isinstance(source, bytes):
        raise TypeError('source must be bytes, not %s' % (type(source),))
    if len(source) != 2:
        raise ValueError('source must be exactly 2 bytes')
    decode = self._decode64
    try:
        if self.big:
            return decode(source[1]) + (decode(source[0]) << 6)
        else:
            return decode(source[0]) + (decode(source[1]) << 6)
    except KeyError:
        raise ValueError('invalid character')