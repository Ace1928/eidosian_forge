from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import urlencode
from .. import errors
from .base import BaseEndpoint
def create_request_token(self, request, credentials):
    """Create and save a new request token.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :param credentials: A dict of extra token credentials.
        :returns: The token as an urlencoded string.
        """
    token = {'oauth_token': self.token_generator(), 'oauth_token_secret': self.token_generator(), 'oauth_callback_confirmed': 'true'}
    token.update(credentials)
    self.request_validator.save_request_token(token, request)
    return urlencode(token.items())