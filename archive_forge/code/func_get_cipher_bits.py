import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_cipher_bits(self):
    """
        Obtain the number of secret bits of the currently used cipher.

        :returns: The number of secret bits of the currently used cipher
            or :obj:`None` if no connection has been established.
        :rtype: :class:`int` or :class:`NoneType`

        .. versionadded:: 0.15
        """
    cipher = _lib.SSL_get_current_cipher(self._ssl)
    if cipher == _ffi.NULL:
        return None
    else:
        return _lib.SSL_CIPHER_get_bits(cipher, _ffi.NULL)