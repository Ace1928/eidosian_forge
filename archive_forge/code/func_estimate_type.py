from __future__ import absolute_import, unicode_literals
import hashlib
import hmac
from binascii import b2a_base64
import warnings
from oauthlib import common
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from . import utils
def estimate_type(self, request):
    """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
    if request.headers.get('Authorization', '').split(' ')[0] == 'Bearer':
        return 9
    elif request.access_token is not None:
        return 5
    else:
        return 0