import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_tlsext_servername_callback(self, callback):
    """
        Specify a callback function to be called when clients specify a server
        name.

        :param callback: The callback function.  It will be invoked with one
            argument, the Connection instance.

        .. versionadded:: 0.13
        """

    @wraps(callback)
    def wrapper(ssl, alert, arg):
        callback(Connection._reverse_mapping[ssl])
        return 0
    self._tlsext_servername_callback = _ffi.callback('int (*)(SSL *, int *, void *)', wrapper)
    _lib.SSL_CTX_set_tlsext_servername_callback(self._context, self._tlsext_servername_callback)