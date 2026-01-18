from __future__ import absolute_import, division, print_function
import base64
import hashlib
import json
import re
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
def create_key_authorization(client, token):
    """
    Returns the key authorization for the given token
    https://tools.ietf.org/html/rfc8555#section-8.1
    """
    accountkey_json = json.dumps(client.account_jwk, sort_keys=True, separators=(',', ':'))
    thumbprint = nopad_b64(hashlib.sha256(accountkey_json.encode('utf8')).digest())
    return '{0}.{1}'.format(token, thumbprint)