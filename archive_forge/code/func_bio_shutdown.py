import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def bio_shutdown(self):
    """
        If the Connection was created with a memory BIO, this method can be
        used to indicate that *end of file* has been reached on the read end of
        that memory BIO.

        :return: None
        """
    if self._from_ssl is None:
        raise TypeError('Connection sock was not None')
    _lib.BIO_set_mem_eof_return(self._into_ssl, 0)