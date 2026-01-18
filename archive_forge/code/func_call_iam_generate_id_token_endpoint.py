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
def call_iam_generate_id_token_endpoint(request, signer_email, audience, access_token):
    """Call iam.generateIdToken endpoint to get ID token.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        signer_email (str): The signer email used to form the IAM
            generateIdToken endpoint.
        audience (str): The audience for the ID token.
        access_token (str): The access token used to call the IAM endpoint.

    Returns:
        Tuple[str, datetime]: The ID token and expiration.
    """
    body = {'audience': audience, 'includeEmail': 'true', 'useEmailAzp': 'true'}
    response_data = _token_endpoint_request(request, _IAM_IDTOKEN_ENDPOINT.format(signer_email), body, access_token=access_token, use_json=True)
    try:
        id_token = response_data['token']
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError('No ID token in response.', response_data, retryable=False)
        six.raise_from(new_exc, caught_exc)
    payload = jwt.decode(id_token, verify=False)
    expiry = datetime.datetime.utcfromtimestamp(payload['exp'])
    return (id_token, expiry)