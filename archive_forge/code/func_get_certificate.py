import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_certificate(self):
    """
        Retrieve the local certificate (if any)

        :return: The local certificate
        """
    cert = _lib.SSL_get_certificate(self._ssl)
    if cert != _ffi.NULL:
        _lib.X509_up_ref(cert)
        return X509._from_raw_x509_ptr(cert)
    return None