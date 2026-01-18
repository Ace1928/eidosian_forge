from __future__ import absolute_import, unicode_literals
import logging
def rotate_refresh_token(self, request):
    """Determine whether to rotate the refresh token.

    Default, yes.

        When access tokens are refreshed the old refresh token can be kept
        or replaced with a new one (rotated). Return True to rotate and
        and False for keeping original.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Refresh Token Grant
        """
    return True