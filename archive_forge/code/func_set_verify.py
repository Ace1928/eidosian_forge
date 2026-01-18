import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_verify(self, mode, callback=None):
    """
        Override the Context object's verification flags for this specific
        connection. See :py:meth:`Context.set_verify` for details.
        """
    if not isinstance(mode, int):
        raise TypeError('mode must be an integer')
    if callback is None:
        self._verify_helper = None
        self._verify_callback = None
        _lib.SSL_set_verify(self._ssl, mode, _ffi.NULL)
    else:
        if not callable(callback):
            raise TypeError('callback must be callable')
        self._verify_helper = _VerifyHelper(callback)
        self._verify_callback = self._verify_helper.callback
        _lib.SSL_set_verify(self._ssl, mode, self._verify_callback)