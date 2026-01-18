import base64
import functools
import hashlib
import hmac
import math
import os
from keystonemiddleware.i18n import _
from oslo_utils import secretutils
def protect_data(keys, data):
    """Serialize data given a dict of keys.

    Given keys and serialized data, returns an appropriately protected string
    suitable for storage in the cache.

    """
    if keys['strategy'] == b'ENCRYPT':
        data = encrypt_data(keys['ENCRYPTION'], data)
    encoded_data = base64.b64encode(data)
    signature = sign_data(keys['MAC'], encoded_data)
    return signature + encoded_data