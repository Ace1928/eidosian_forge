from __future__ import absolute_import, unicode_literals
import logging
def revoke_token(self, token, token_type_hint, request, *args, **kwargs):
    """Revoke an access or refresh token.

        :param token: The token string.
        :param token_type_hint: access_token or refresh_token.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        Method is used by:
            - Revocation Endpoint
        """
    raise NotImplementedError('Subclasses must implement this method.')