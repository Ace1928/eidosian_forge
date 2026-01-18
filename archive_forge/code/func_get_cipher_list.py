import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_cipher_list(self):
    """
        Retrieve the list of ciphers used by the Connection object.

        :return: A list of native cipher strings.
        """
    ciphers = []
    for i in count():
        result = _lib.SSL_get_cipher_list(self._ssl, i)
        if result == _ffi.NULL:
            break
        ciphers.append(_ffi.string(result).decode('utf-8'))
    return ciphers