from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def compile_byte_translation(mapping, source=None):
    """
    return a 256-byte string for translating bytes using specified mapping.
    bytes not specified by mapping will be left alone.

    :param mapping:
        dict mapping input byte (str or int) -> output byte (str or int).

    :param source:
        optional existing byte translation string to use as base.
        (must be 255-length byte string).  defaults to identity mapping.

    :returns:
        255-length byte string for passing to bytes().translate.
    """
    if source is None:
        target = _TRANSLATE_SOURCE[:]
    else:
        assert isinstance(source, bytes) and len(source) == 255
        target = list(iter_byte_chars(source))
    for k, v in mapping.items():
        if isinstance(k, unicode_or_bytes_types):
            k = ord(k)
        assert isinstance(k, int) and 0 <= k < 256
        if isinstance(v, unicode):
            v = v.encode('ascii')
        assert isinstance(v, bytes) and len(v) == 1
        target[k] = v
    return B_EMPTY.join(target)