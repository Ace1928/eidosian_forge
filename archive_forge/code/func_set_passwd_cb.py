import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_passwd_cb(self, callback, userdata=None):
    """
        Set the passphrase callback.  This function will be called
        when a private key with a passphrase is loaded.

        :param callback: The Python callback to use.  This must accept three
            positional arguments.  First, an integer giving the maximum length
            of the passphrase it may return.  If the returned passphrase is
            longer than this, it will be truncated.  Second, a boolean value
            which will be true if the user should be prompted for the
            passphrase twice and the callback should verify that the two values
            supplied are equal. Third, the value given as the *userdata*
            parameter to :meth:`set_passwd_cb`.  The *callback* must return
            a byte string. If an error occurs, *callback* should return a false
            value (e.g. an empty string).
        :param userdata: (optional) A Python object which will be given as
                         argument to the callback
        :return: None
        """
    if not callable(callback):
        raise TypeError('callback must be callable')
    self._passphrase_helper = self._wrap_callback(callback)
    self._passphrase_callback = self._passphrase_helper.callback
    _lib.SSL_CTX_set_default_passwd_cb(self._context, self._passphrase_callback)
    self._passphrase_userdata = userdata