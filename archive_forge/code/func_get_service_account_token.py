import datetime
import json
import logging
import os
import six
from six.moves import http_client
from six.moves.urllib import parse as urlparse
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
def get_service_account_token(request, service_account='default', scopes=None):
    """Get the OAuth 2.0 access token for a service account.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        service_account (str): The string 'default' or a service account email
            address. The determines which service account for which to acquire
            an access token.
        scopes (Optional[Union[str, List[str]]]): Optional string or list of
            strings with auth scopes.
    Returns:
        Tuple[str, datetime]: The access token and its expiration.

    Raises:
        google.auth.exceptions.TransportError: if an error occurred while
            retrieving metadata.
    """
    if scopes:
        if not isinstance(scopes, str):
            scopes = ','.join(scopes)
        params = {'scopes': scopes}
    else:
        params = None
    path = 'instance/service-accounts/{0}/token'.format(service_account)
    token_json = get(request, path, params=params)
    token_expiry = _helpers.utcnow() + datetime.timedelta(seconds=token_json['expires_in'])
    return (token_json['access_token'], token_expiry)