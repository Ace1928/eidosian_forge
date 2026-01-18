import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
@_helpers.positional(1)
def step1_get_authorize_url(self, redirect_uri=None, state=None):
    """Returns a URI to redirect to the provider.

        Args:
            redirect_uri: string, Either the string 'urn:ietf:wg:oauth:2.0:oob'
                          for a non-web-based application, or a URI that
                          handles the callback from the authorization server.
                          This parameter is deprecated, please move to passing
                          the redirect_uri in via the constructor.
            state: string, Opaque state string which is passed through the
                   OAuth2 flow and returned to the client as a query parameter
                   in the callback.

        Returns:
            A URI as a string to redirect the user to begin the authorization
            flow.
        """
    if redirect_uri is not None:
        logger.warning('The redirect_uri parameter for OAuth2WebServerFlow.step1_get_authorize_url is deprecated. Please move to passing the redirect_uri in via the constructor.')
        self.redirect_uri = redirect_uri
    if self.redirect_uri is None:
        raise ValueError('The value of redirect_uri must not be None.')
    query_params = {'client_id': self.client_id, 'redirect_uri': self.redirect_uri, 'scope': self.scope}
    if state is not None:
        query_params['state'] = state
    if self.login_hint is not None:
        query_params['login_hint'] = self.login_hint
    if self._pkce:
        if not self.code_verifier:
            self.code_verifier = _pkce.code_verifier()
        challenge = _pkce.code_challenge(self.code_verifier)
        query_params['code_challenge'] = challenge
        query_params['code_challenge_method'] = 'S256'
    query_params.update(self.params)
    return _helpers.update_query_params(self.auth_uri, query_params)