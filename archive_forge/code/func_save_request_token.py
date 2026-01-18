from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def save_request_token(self, token, request):
    """Save an OAuth1 request token.

        :param token: A dict with token credentials.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        The token dictionary will at minimum include

        * ``oauth_token`` the request token string.
        * ``oauth_token_secret`` the token specific secret used in signing.
        * ``oauth_callback_confirmed`` the string ``true``.

        Client key can be obtained from ``request.client_key``.

        This method is used by

        * RequestTokenEndpoint
        """
    raise self._subclass_must_implement('save_request_token')