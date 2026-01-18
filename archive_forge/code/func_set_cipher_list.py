import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_cipher_list(self, cipher_list):
    """
        Set the list of ciphers to be used in this context.

        See the OpenSSL manual for more information (e.g.
        :manpage:`ciphers(1)`).

        :param bytes cipher_list: An OpenSSL cipher string.
        :return: None
        """
    cipher_list = _text_to_bytes_and_warn('cipher_list', cipher_list)
    if not isinstance(cipher_list, bytes):
        raise TypeError('cipher_list must be a byte string.')
    _openssl_assert(_lib.SSL_CTX_set_cipher_list(self._context, cipher_list) == 1)
    tmpconn = Connection(self, None)
    if tmpconn.get_cipher_list() == ['TLS_AES_256_GCM_SHA384', 'TLS_CHACHA20_POLY1305_SHA256', 'TLS_AES_128_GCM_SHA256']:
        raise Error([('SSL routines', 'SSL_CTX_set_cipher_list', 'no cipher match')])