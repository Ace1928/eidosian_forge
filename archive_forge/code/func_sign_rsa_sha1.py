from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def sign_rsa_sha1(base_string, rsa_private_key):
    """**RSA-SHA1**

    Per `section 3.4.3`_ of the spec.

    The "RSA-SHA1" signature method uses the RSASSA-PKCS1-v1_5 signature
    algorithm as defined in `RFC3447, Section 8.2`_ (also known as
    PKCS#1), using SHA-1 as the hash function for EMSA-PKCS1-v1_5.  To
    use this method, the client MUST have established client credentials
    with the server that included its RSA public key (in a manner that is
    beyond the scope of this specification).

    .. _`section 3.4.3`: https://tools.ietf.org/html/rfc5849#section-3.4.3
    .. _`RFC3447, Section 8.2`: https://tools.ietf.org/html/rfc3447#section-8.2

    """
    if isinstance(base_string, unicode_type):
        base_string = base_string.encode('utf-8')
    alg = _jwt_rs1_signing_algorithm()
    key = _prepare_key_plus(alg, rsa_private_key)
    s = alg.sign(base_string, key)
    return binascii.b2a_base64(s)[:-1].decode('utf-8')