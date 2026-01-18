import json
from six.moves import urllib
from google_reauth import errors
def refresh_grant(http_request, client_id, client_secret, refresh_token, token_uri, scopes=None, rapt=None, headers={}):
    """Implements the OAuth 2.0 Refresh Grant with the addition of the reauth
    token.

    Args:
        http_request (Callable): callable to run http requests. Accepts uri,
            method, body and headers. Returns a tuple: (response, content)
        client_id (str): client id to get access token for reauth scope.
        client_secret (str): client secret for the client_id
        refresh_token (str): refresh token to refresh access token
        token_uri (str): uri to refresh access token
        scopes (str): scopes required by the client application as a
            comma-joined list.
        rapt (str): RAPT token
        headers (dict): headers for http request

    Returns:
        Tuple[str, dict]: http response and parsed response content.
    """
    parameters = {'grant_type': 'refresh_token', 'client_id': client_id, 'client_secret': client_secret, 'refresh_token': refresh_token}
    if scopes:
        parameters['scope'] = scopes
    if rapt:
        parameters['rapt'] = rapt
    body = urllib.parse.urlencode(parameters)
    response, content = http_request(uri=token_uri, method='POST', body=body, headers=headers)
    return (response, content)