import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def bio_write(self, buf):
    """
        If the Connection was created with a memory BIO, this method can be
        used to add bytes to the read end of that memory BIO.  The Connection
        can then read the bytes (for example, in response to a call to
        :meth:`recv`).

        :param buf: The string to put into the memory BIO.
        :return: The number of bytes written
        """
    buf = _text_to_bytes_and_warn('buf', buf)
    if self._into_ssl is None:
        raise TypeError('Connection sock was not None')
    with _ffi.from_buffer(buf) as data:
        result = _lib.BIO_write(self._into_ssl, data, len(data))
        if result <= 0:
            self._handle_bio_errors(self._into_ssl, result)
        return result