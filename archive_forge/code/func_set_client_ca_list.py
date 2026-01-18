import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_client_ca_list(self, certificate_authorities):
    """
        Set the list of preferred client certificate signers for this server
        context.

        This list of certificate authorities will be sent to the client when
        the server requests a client certificate.

        :param certificate_authorities: a sequence of X509Names.
        :return: None

        .. versionadded:: 0.10
        """
    name_stack = _lib.sk_X509_NAME_new_null()
    _openssl_assert(name_stack != _ffi.NULL)
    try:
        for ca_name in certificate_authorities:
            if not isinstance(ca_name, X509Name):
                raise TypeError('client CAs must be X509Name objects, not %s objects' % (type(ca_name).__name__,))
            copy = _lib.X509_NAME_dup(ca_name._name)
            _openssl_assert(copy != _ffi.NULL)
            push_result = _lib.sk_X509_NAME_push(name_stack, copy)
            if not push_result:
                _lib.X509_NAME_free(copy)
                _raise_current_error()
    except Exception:
        _lib.sk_X509_NAME_free(name_stack)
        raise
    _lib.SSL_CTX_set_client_CA_list(self._context, name_stack)