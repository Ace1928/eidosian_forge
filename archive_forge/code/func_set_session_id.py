import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_session_id(self, buf):
    """
        Set the session id to *buf* within which a session can be reused for
        this Context object.  This is needed when doing session resumption,
        because there is no way for a stored session to know which Context
        object it is associated with.

        :param bytes buf: The session id.

        :returns: None
        """
    buf = _text_to_bytes_and_warn('buf', buf)
    _openssl_assert(_lib.SSL_CTX_set_session_id_context(self._context, buf, len(buf)) == 1)