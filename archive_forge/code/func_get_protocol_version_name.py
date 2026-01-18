import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_protocol_version_name(self):
    """
        Retrieve the protocol version of the current connection.

        :returns: The TLS version of the current connection, for example
            the value for TLS 1.2 would be ``TLSv1.2``or ``Unknown``
            for connections that were not successfully established.
        :rtype: :class:`unicode`
        """
    version = _ffi.string(_lib.SSL_get_version(self._ssl))
    return version.decode('utf-8')