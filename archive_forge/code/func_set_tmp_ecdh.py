import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_tmp_ecdh(self, curve):
    """
        Select a curve to use for ECDHE key exchange.

        :param curve: A curve object to use as returned by either
            :meth:`OpenSSL.crypto.get_elliptic_curve` or
            :meth:`OpenSSL.crypto.get_elliptic_curves`.

        :return: None
        """
    _lib.SSL_CTX_set_tmp_ecdh(self._context, curve._to_EC_KEY())