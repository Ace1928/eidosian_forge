import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def bio_read(self, bufsiz):
    """
        If the Connection was created with a memory BIO, this method can be
        used to read bytes from the write end of that memory BIO.  Many
        Connection methods will add bytes which must be read in this manner or
        the buffer will eventually fill up and the Connection will be able to
        take no further actions.

        :param bufsiz: The maximum number of bytes to read
        :return: The string read.
        """
    if self._from_ssl is None:
        raise TypeError('Connection sock was not None')
    if not isinstance(bufsiz, int):
        raise TypeError('bufsiz must be an integer')
    buf = _no_zero_allocator('char[]', bufsiz)
    result = _lib.BIO_read(self._from_ssl, buf, bufsiz)
    if result <= 0:
        self._handle_bio_errors(self._from_ssl, result)
    return _ffi.buffer(buf, result)[:]