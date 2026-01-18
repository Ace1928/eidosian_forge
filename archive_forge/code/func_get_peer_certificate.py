import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_peer_certificate(self):
    """
        Retrieve the other side's certificate (if any)

        :return: The peer's certificate
        """
    cert = _lib.SSL_get_peer_certificate(self._ssl)
    if cert != _ffi.NULL:
        return X509._from_raw_x509_ptr(cert)
    return None