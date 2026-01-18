import base64
import functools
import hashlib
import hmac
import math
import os
from keystonemiddleware.i18n import _
from oslo_utils import secretutils
def assert_crypto_availability(f):
    """Ensure cryptography module is available."""

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        if ciphers is None:
            raise CryptoUnavailableError()
        return f(*args, **kwds)
    return wrapper