import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_session_cache_mode(self):
    """
        Get the current session cache mode.

        :returns: The currently used cache mode.

        .. versionadded:: 0.14
        """
    return _lib.SSL_CTX_get_session_cache_mode(self._context)