from __future__ import absolute_import, unicode_literals
import logging
def is_within_original_scope(self, request_scopes, refresh_token, request, *args, **kwargs):
    """Check if requested scopes are within a scope of the refresh token.

        When access tokens are refreshed the scope of the new token
        needs to be within the scope of the original token. This is
        ensured by checking that all requested scopes strings are on
        the list returned by the get_original_scopes. If this check
        fails, is_within_original_scope is called. The method can be
        used in situations where returning all valid scopes from the
        get_original_scopes is not practical.

        :param request_scopes: A list of scopes that were requested by client.
        :param refresh_token: Unicode refresh_token.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Refresh token grant
        """
    return False