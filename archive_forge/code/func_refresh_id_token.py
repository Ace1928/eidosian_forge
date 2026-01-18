import logging
from oauthlib.oauth2.rfc6749.request_validator import (
def refresh_id_token(self, request):
    """Whether the id token should be refreshed. Default, True

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            RefreshTokenGrant
        """
    return True