import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def load_tmp_dh(self, dhfile):
    """
        Load parameters for Ephemeral Diffie-Hellman

        :param dhfile: The file to load EDH parameters from (``bytes`` or
            ``unicode``).

        :return: None
        """
    dhfile = _path_bytes(dhfile)
    bio = _lib.BIO_new_file(dhfile, b'r')
    if bio == _ffi.NULL:
        _raise_current_error()
    bio = _ffi.gc(bio, _lib.BIO_free)
    dh = _lib.PEM_read_bio_DHparams(bio, _ffi.NULL, _ffi.NULL, _ffi.NULL)
    dh = _ffi.gc(dh, _lib.DH_free)
    res = _lib.SSL_CTX_set_tmp_dh(self._context, dh)
    _openssl_assert(res == 1)