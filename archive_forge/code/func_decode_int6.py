from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def decode_int6(self, source):
    """decode single character -> 6 bit integer"""
    if not isinstance(source, bytes):
        raise TypeError('source must be bytes, not %s' % (type(source),))
    if len(source) != 1:
        raise ValueError('source must be exactly 1 byte')
    if PY3:
        source = source[0]
    try:
        return self._decode64(source)
    except KeyError:
        raise ValueError('invalid character')