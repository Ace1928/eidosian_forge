import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_verify_depth(self):
    """
        Retrieve the Context object's verify depth, as set by
        :meth:`set_verify_depth`.

        :return: The verify depth
        """
    return _lib.SSL_CTX_get_verify_depth(self._context)