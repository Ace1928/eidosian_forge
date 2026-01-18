import base64
import functools
import hashlib
import hmac
import math
import os
from keystonemiddleware.i18n import _
from oslo_utils import secretutils
def sign_data(key, data):
    """Sign the data using the defined function and the derived key."""
    if not isinstance(key, bytes):
        key = key.encode()
    if not isinstance(data, bytes):
        data = data.encode()
    mac = hmac.new(key, data, HASH_FUNCTION).digest()
    return base64.b64encode(mac)