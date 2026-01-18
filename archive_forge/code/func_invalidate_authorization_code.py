from __future__ import absolute_import, unicode_literals
import logging
def invalidate_authorization_code(self, client_id, code, request, *args, **kwargs):
    """Invalidate an authorization code after use.

        :param client_id: Unicode client identifier.
        :param code: The authorization code grant (request.code).
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        Method is used by:
            - Authorization Code Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')