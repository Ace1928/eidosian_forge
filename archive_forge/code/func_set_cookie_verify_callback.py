import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_cookie_verify_callback(self, callback):
    self._cookie_verify_helper = _CookieVerifyCallbackHelper(callback)
    _lib.SSL_CTX_set_cookie_verify_cb(self._context, self._cookie_verify_helper.callback)