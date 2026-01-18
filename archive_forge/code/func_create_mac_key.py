from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import re
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64
def create_mac_key(self, alg, key):
    """Create a MAC key."""
    if alg == 'HS256':
        hashalg = 'sha256'
        hashbytes = 32
    elif alg == 'HS384':
        hashalg = 'sha384'
        hashbytes = 48
    elif alg == 'HS512':
        hashalg = 'sha512'
        hashbytes = 64
    else:
        raise BackendException('Unsupported MAC key algorithm for OpenSSL backend: {0}'.format(alg))
    key_bytes = base64.urlsafe_b64decode(key)
    if len(key_bytes) < hashbytes:
        raise BackendException('{0} key must be at least {1} bytes long (after Base64 decoding)'.format(alg, hashbytes))
    return {'type': 'hmac', 'alg': alg, 'jwk': {'kty': 'oct', 'k': key}, 'hash': hashalg}