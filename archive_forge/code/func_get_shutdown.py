import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_shutdown(self):
    """
        Get the shutdown state of the Connection.

        :return: The shutdown state, a bitvector of SENT_SHUTDOWN,
            RECEIVED_SHUTDOWN.
        """
    return _lib.SSL_get_shutdown(self._ssl)