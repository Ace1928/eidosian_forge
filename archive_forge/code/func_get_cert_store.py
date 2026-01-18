import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_cert_store(self):
    """
        Get the certificate store for the context.  This can be used to add
        "trusted" certificates without using the
        :meth:`load_verify_locations` method.

        :return: A X509Store object or None if it does not have one.
        """
    store = _lib.SSL_CTX_get_cert_store(self._context)
    if store == _ffi.NULL:
        return None
    pystore = X509Store.__new__(X509Store)
    pystore._store = store
    return pystore