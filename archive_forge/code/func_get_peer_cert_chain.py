import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_peer_cert_chain(self):
    """
        Retrieve the other side's certificate (if any)

        :return: A list of X509 instances giving the peer's certificate chain,
                 or None if it does not have one.
        """
    cert_stack = _lib.SSL_get_peer_cert_chain(self._ssl)
    if cert_stack == _ffi.NULL:
        return None
    return self._cert_stack_to_list(cert_stack)