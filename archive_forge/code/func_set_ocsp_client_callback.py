import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_ocsp_client_callback(self, callback, data=None):
    """
        Set a callback to validate OCSP data stapled to the TLS handshake on
        the client side.

        :param callback: The callback function. It will be invoked with three
            arguments: the Connection, a bytestring containing the stapled OCSP
            assertion, and the optional arbitrary data you have provided. The
            callback must return a boolean that indicates the result of
            validating the OCSP data: ``True`` if the OCSP data is valid and
            the certificate can be trusted, or ``False`` if either the OCSP
            data is invalid or the certificate has been revoked.
        :param data: Some opaque data that will be passed into the callback
            function when called. This can be used to avoid needing to do
            complex data lookups or to keep track of what context is being
            used. This parameter is optional.
        """
    helper = _OCSPClientCallbackHelper(callback)
    self._set_ocsp_callback(helper, data)