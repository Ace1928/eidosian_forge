from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def encode_int30(self, value):
    """decode 5 char string -> 30 bit integer"""
    if value < 0 or value > 1073741823:
        raise ValueError('value out of range')
    return self._encode_int(value, 30)