from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def decode_transposed_bytes(self, source, offsets):
    """decode byte string, then reverse transposition described by offset list"""
    tmp = self.decode_bytes(source)
    buf = [None] * len(offsets)
    for off, char in zip(offsets, tmp):
        buf[off] = char
    return join_byte_elems(buf)