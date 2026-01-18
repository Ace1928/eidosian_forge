from __future__ import absolute_import, unicode_literals
import logging
def get_jwt_bearer_token(self, token, token_handler, request):
    """Get JWT Bearer token or OpenID Connect ID token

        If using OpenID Connect this SHOULD call
        `oauthlib.oauth2.RequestValidator.get_id_token`

        :param token: A Bearer token dict.
        :param token_handler: The token handler (BearerToken class).
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :return: The JWT Bearer token or OpenID Connect ID token (a JWS signed
        JWT)

        Method is used by JWT Bearer and OpenID Connect tokens:
            - JWTToken.create_token
        """
    raise NotImplementedError('Subclasses must implement this method.')