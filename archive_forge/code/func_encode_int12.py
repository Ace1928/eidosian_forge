from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def encode_int12(self, value):
    """encodes 12-bit integer -> 2 char string"""
    if value < 0 or value > 4095:
        raise ValueError('value out of range')
    raw = [value & 63, value >> 6 & 63]
    if self.big:
        raw = reversed(raw)
    return join_byte_elems(imap(self._encode64, raw))