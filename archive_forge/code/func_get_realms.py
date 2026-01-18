from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def get_realms(self, token, request):
    """Get realms associated with a request token.

        :param token: The request token string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: The list of realms associated with the request token.

        This method is used by

        * AuthorizationEndpoint
        * AccessTokenEndpoint
        """
    raise self._subclass_must_implement('get_realms')