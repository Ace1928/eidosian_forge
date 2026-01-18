from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def save_verifier(self, token, verifier, request):
    """Associate an authorization verifier with a request token.

        :param token: A request token string.
        :param verifier A dictionary containing the oauth_verifier and
                        oauth_token
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        We need to associate verifiers with tokens for validation during the
        access token request.

        Note that unlike save_x_token token here is the ``oauth_token`` token
        string from the request token saved previously.

        This method is used by

        * AuthorizationEndpoint
        """
    raise self._subclass_must_implement('save_verifier')