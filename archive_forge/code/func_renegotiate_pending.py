import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def renegotiate_pending(self):
    """
        Check if there's a renegotiation in progress, it will return False once
        a renegotiation is finished.

        :return: Whether there's a renegotiation in progress
        :rtype: bool
        """
    return _lib.SSL_renegotiate_pending(self._ssl) == 1