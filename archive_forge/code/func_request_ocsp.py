import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def request_ocsp(self):
    """
        Called to request that the server sends stapled OCSP data, if
        available. If this is not called on the client side then the server
        will not send OCSP data. Should be used in conjunction with
        :meth:`Context.set_ocsp_client_callback`.
        """
    rc = _lib.SSL_set_tlsext_status_type(self._ssl, _lib.TLSEXT_STATUSTYPE_ocsp)
    _openssl_assert(rc == 1)