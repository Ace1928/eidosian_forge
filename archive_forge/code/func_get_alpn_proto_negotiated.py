import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
@_requires_alpn
def get_alpn_proto_negotiated(self):
    """
        Get the protocol that was negotiated by ALPN.

        :returns: A bytestring of the protocol name.  If no protocol has been
            negotiated yet, returns an empty bytestring.
        """
    data = _ffi.new('unsigned char **')
    data_len = _ffi.new('unsigned int *')
    _lib.SSL_get0_alpn_selected(self._ssl, data, data_len)
    if not data_len:
        return b''
    return _ffi.buffer(data[0], data_len[0])[:]