from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def encode_bytes(self, source):
    """encode bytes to base64 string.

        :arg source: byte string to encode.
        :returns: byte string containing encoded data.
        """
    if not isinstance(source, bytes):
        raise TypeError('source must be bytes, not %s' % (type(source),))
    chunks, tail = divmod(len(source), 3)
    if PY3:
        next_value = nextgetter(iter(source))
    else:
        next_value = nextgetter((ord(elem) for elem in source))
    gen = self._encode_bytes(next_value, chunks, tail)
    out = join_byte_elems(imap(self._encode64, gen))
    return out