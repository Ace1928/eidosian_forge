import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_info_callback(self, callback):
    """
        Set the information callback to *callback*. This function will be
        called from time to time during SSL handshakes.

        :param callback: The Python callback to use.  This should take three
            arguments: a Connection object and two integers.  The first integer
            specifies where in the SSL handshake the function was called, and
            the other the return code from a (possibly failed) internal
            function call.
        :return: None
        """

    @wraps(callback)
    def wrapper(ssl, where, return_code):
        callback(Connection._reverse_mapping[ssl], where, return_code)
    self._info_callback = _ffi.callback('void (*)(const SSL *, int, int)', wrapper)
    _lib.SSL_CTX_set_info_callback(self._context, self._info_callback)