import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_ocsp_server_callback(self, callback, data=None):
    """
        Set a callback to provide OCSP data to be stapled to the TLS handshake
        on the server side.

        :param callback: The callback function. It will be invoked with two
            arguments: the Connection, and the optional arbitrary data you have
            provided. The callback must return a bytestring that contains the
            OCSP data to staple to the handshake. If no OCSP data is available
            for this connection, return the empty bytestring.
        :param data: Some opaque data that will be passed into the callback
            function when called. This can be used to avoid needing to do
            complex data lookups or to keep track of what context is being
            used. This parameter is optional.
        """
    helper = _OCSPServerCallbackHelper(callback)
    self._set_ocsp_callback(helper, data)