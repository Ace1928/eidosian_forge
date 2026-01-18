import datetime
import json
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _exponential_backoff
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
def id_token_jwt_grant(request, token_uri, assertion, can_retry=True):
    """Implements the JWT Profile for OAuth 2.0 Authorization Grants, but
    requests an OpenID Connect ID Token instead of an access token.

    This is a variant on the standard JWT Profile that is currently unique
    to Google. This was added for the benefit of authenticating to services
    that require ID Tokens instead of access tokens or JWT bearer tokens.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorization server's token endpoint
            URI.
        assertion (str): JWT token signed by a service account. The token's
            payload must include a ``target_audience`` claim.
        can_retry (bool): Enable or disable request retry behavior.

    Returns:
        Tuple[str, Optional[datetime], Mapping[str, str]]:
            The (encoded) Open ID Connect ID Token, expiration, and additional
            data returned by the endpoint.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.
    """
    body = {'assertion': assertion, 'grant_type': _JWT_GRANT_TYPE}
    response_data = _token_endpoint_request(request, token_uri, body, can_retry=can_retry)
    try:
        id_token = response_data['id_token']
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError('No ID token in response.', response_data, retryable=False)
        six.raise_from(new_exc, caught_exc)
    payload = jwt.decode(id_token, verify=False)
    expiry = datetime.datetime.utcfromtimestamp(payload['exp'])
    return (id_token, expiry, response_data)