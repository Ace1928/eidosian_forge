from __future__ import absolute_import, unicode_literals
import logging
def get_default_scopes(self, client_id, request, *args, **kwargs):
    """Get the default scopes for the client.

        :param client_id: Unicode client identifier.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: List of default scopes

        Method is used by all core grant types:
            - Authorization Code Grant
            - Implicit Grant
            - Resource Owner Password Credentials Grant
            - Client Credentials grant
        """
    raise NotImplementedError('Subclasses must implement this method.')