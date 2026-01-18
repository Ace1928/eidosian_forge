from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def sign_hmac_sha256(base_string, client_secret, resource_owner_secret):
    """**HMAC-SHA256**

    The "HMAC-SHA256" signature method uses the HMAC-SHA256 signature
    algorithm as defined in `RFC4634`_::

        digest = HMAC-SHA256 (key, text)

    Per `section 3.4.2`_ of the spec.

    .. _`RFC4634`: https://tools.ietf.org/html/rfc4634
    .. _`section 3.4.2`: https://tools.ietf.org/html/rfc5849#section-3.4.2
    """
    text = base_string
    key = utils.escape(client_secret or '')
    key += '&'
    key += utils.escape(resource_owner_secret or '')
    key_utf8 = key.encode('utf-8')
    text_utf8 = text.encode('utf-8')
    signature = hmac.new(key_utf8, text_utf8, hashlib.sha256)
    return binascii.b2a_base64(signature.digest())[:-1].decode('utf-8')