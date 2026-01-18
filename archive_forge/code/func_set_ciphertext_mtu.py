import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_ciphertext_mtu(self, mtu):
    """
        For DTLS, set the maximum UDP payload size (*not* including IP/UDP
        overhead).

        Note that you might have to set :data:`OP_NO_QUERY_MTU` to prevent
        OpenSSL from spontaneously clearing this.

        :param mtu: An integer giving the maximum transmission unit.

        .. versionadded:: 21.1
        """
    _lib.SSL_set_mtu(self._ssl, mtu)