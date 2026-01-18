from __future__ import absolute_import, unicode_literals
import logging
def get_original_scopes(self, refresh_token, request, *args, **kwargs):
    """Get the list of scopes associated with the refresh token.

        :param refresh_token: Unicode refresh token.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: List of scopes.

        Method is used by:
            - Refresh token grant
        """
    raise NotImplementedError('Subclasses must implement this method.')