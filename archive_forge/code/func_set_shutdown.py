import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_shutdown(self, state):
    """
        Set the shutdown state of the Connection.

        :param state: bitvector of SENT_SHUTDOWN, RECEIVED_SHUTDOWN.
        :return: None
        """
    if not isinstance(state, int):
        raise TypeError('state must be an integer')
    _lib.SSL_set_shutdown(self._ssl, state)