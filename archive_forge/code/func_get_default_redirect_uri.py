from __future__ import absolute_import, unicode_literals
import logging
def get_default_redirect_uri(self, client_id, request, *args, **kwargs):
    """Get the default redirect URI for the client.

        :param client_id: Unicode client identifier.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: The default redirect URI for the client

        Method is used by:
            - Authorization Code Grant
            - Implicit Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')