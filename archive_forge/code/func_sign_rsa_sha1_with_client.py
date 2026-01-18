from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def sign_rsa_sha1_with_client(base_string, client):
    if not client.rsa_key:
        raise ValueError('rsa_key is required when using RSA signature method.')
    return sign_rsa_sha1(base_string, client.rsa_key)