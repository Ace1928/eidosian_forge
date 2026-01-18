import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def sock_shutdown(self, *args, **kwargs):
    """
        Call the :meth:`shutdown` method of the underlying socket.
        See :manpage:`shutdown(2)`.

        :return: What the socket's shutdown() method returns
        """
    return self._socket.shutdown(*args, **kwargs)