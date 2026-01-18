import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_session_cache_mode(self, mode):
    """
        Set the behavior of the session cache used by all connections using
        this Context.  The previously set mode is returned.  See
        :const:`SESS_CACHE_*` for details about particular modes.

        :param mode: One or more of the SESS_CACHE_* flags (combine using
            bitwise or)
        :returns: The previously set caching mode.

        .. versionadded:: 0.14
        """
    if not isinstance(mode, int):
        raise TypeError('mode must be an integer')
    return _lib.SSL_CTX_set_session_cache_mode(self._context, mode)