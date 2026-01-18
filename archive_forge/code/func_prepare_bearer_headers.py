from __future__ import absolute_import, unicode_literals
import hashlib
import hmac
from binascii import b2a_base64
import warnings
from oauthlib import common
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from . import utils
def prepare_bearer_headers(token, headers=None):
    """Add a `Bearer Token`_ to the request URI.

    Recommended method of passing bearer tokens.

    Authorization: Bearer h480djs93hd8

    .. _`Bearer Token`: https://tools.ietf.org/html/rfc6750

    :param token:
    :param headers:
    """
    headers = headers or {}
    headers['Authorization'] = 'Bearer %s' % token
    return headers