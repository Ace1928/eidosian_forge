from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from google_reauth import challenges
from google_reauth import errors
from google_reauth import _helpers
from google_reauth import _reauth_client
from six.moves import http_client
from six.moves import range
def get_rapt_token(http_request, client_id, client_secret, refresh_token, token_uri, scopes=None):
    """Given an http request method and refresh_token, get rapt token.

    Args:
        http_request: callable to run http requests. Accepts uri, method, body
            and headers. Returns a tuple: (response, content)
        client_id: client id to get access token for reauth scope.
        client_secret: client secret for the client_id
        refresh_token: refresh token to refresh access token
        token_uri: uri to refresh access token
        scopes: scopes required by the client application

    Returns: rapt token.
    Raises:
        errors.ReauthError if reauth failed
    """
    sys.stderr.write('Reauthentication required.\n')
    response, content = _reauth_client.refresh_grant(http_request=http_request, client_id=client_id, client_secret=client_secret, refresh_token=refresh_token, token_uri=token_uri, scopes=_REAUTH_SCOPE, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    try:
        content = json.loads(content)
    except (TypeError, ValueError):
        raise errors.ReauthAccessTokenRefreshError('Invalid response {0}'.format(_substr_for_error_message(content)))
    if response.status != http_client.OK:
        raise errors.ReauthAccessTokenRefreshError(_get_refresh_error_message(content), response.status)
    if 'access_token' not in content:
        raise errors.ReauthAccessTokenRefreshError('Access token missing from the response')
    rapt_token = _obtain_rapt(http_request, content['access_token'], requested_scopes=scopes)
    return rapt_token