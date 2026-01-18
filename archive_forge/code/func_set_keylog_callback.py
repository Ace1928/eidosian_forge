import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
@_requires_keylog
def set_keylog_callback(self, callback):
    """
        Set the TLS key logging callback to *callback*. This function will be
        called whenever TLS key material is generated or received, in order
        to allow applications to store this keying material for debugging
        purposes.

        :param callback: The Python callback to use.  This should take two
            arguments: a Connection object and a bytestring that contains
            the key material in the format used by NSS for its SSLKEYLOGFILE
            debugging output.
        :return: None
        """

    @wraps(callback)
    def wrapper(ssl, line):
        line = _ffi.string(line)
        callback(Connection._reverse_mapping[ssl], line)
    self._keylog_callback = _ffi.callback('void (*)(const SSL *, const char *)', wrapper)
    _lib.SSL_CTX_set_keylog_callback(self._context, self._keylog_callback)