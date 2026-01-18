from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
@memoized_property
def supported_by_hashlib_pbkdf2(self):
    """helper to detect if hash is supported by hashlib.pbkdf2_hmac()"""
    if not _stdlib_pbkdf2_hmac:
        return None
    try:
        _stdlib_pbkdf2_hmac(self.name, b'p', b's', 1)
        return True
    except ValueError:
        return False