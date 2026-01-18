import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_timeout(self, timeout):
    """
        Set the timeout for newly created sessions for this Context object to
        *timeout*.  The default value is 300 seconds. See the OpenSSL manual
        for more information (e.g. :manpage:`SSL_CTX_set_timeout(3)`).

        :param timeout: The timeout in (whole) seconds
        :return: The previous session timeout
        """
    if not isinstance(timeout, int):
        raise TypeError('timeout must be an integer')
    return _lib.SSL_CTX_set_timeout(self._context, timeout)