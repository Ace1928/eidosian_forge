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
def jwt_grant(request, token_uri, assertion, can_retry=True):
    """Implements the JWT Profile for OAuth 2.0 Authorization Grants.

    For more details, see `rfc7523 section 4`_.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorizations server's token endpoint
            URI.
        assertion (str): The OAuth 2.0 assertion.
        can_retry (bool): Enable or disable request retry behavior.

    Returns:
        Tuple[str, Optional[datetime], Mapping[str, str]]: The access token,
            expiration, and additional data returned by the token endpoint.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.

    .. _rfc7523 section 4: https://tools.ietf.org/html/rfc7523#section-4
    """
    body = {'assertion': assertion, 'grant_type': _JWT_GRANT_TYPE}
    response_data = _token_endpoint_request(request, token_uri, body, can_retry=can_retry)
    try:
        access_token = response_data['access_token']
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError('No access token in response.', response_data, retryable=False)
        six.raise_from(new_exc, caught_exc)
    expiry = _parse_expiry(response_data)
    return (access_token, expiry, response_data)