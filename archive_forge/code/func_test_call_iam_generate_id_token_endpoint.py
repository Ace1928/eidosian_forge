import datetime
import json
import os
import mock
import pytest  # type: ignore
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _client
def test_call_iam_generate_id_token_endpoint():
    now = _helpers.utcnow()
    id_token_expiry = _helpers.datetime_to_secs(now)
    id_token = jwt.encode(SIGNER, {'exp': id_token_expiry}).decode('utf-8')
    request = make_request({'token': id_token})
    token, expiry = _client.call_iam_generate_id_token_endpoint(request, 'fake_email', 'fake_audience', 'fake_access_token')
    assert request.call_args[1]['url'] == 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/fake_email:generateIdToken'
    assert request.call_args[1]['headers']['Content-Type'] == 'application/json'
    assert request.call_args[1]['headers']['Authorization'] == 'Bearer fake_access_token'
    response_body = json.loads(request.call_args[1]['body'])
    assert response_body['audience'] == 'fake_audience'
    assert response_body['includeEmail'] == 'true'
    assert response_body['useEmailAzp'] == 'true'
    assert token == id_token
    now = now.replace(microsecond=0)
    assert expiry == now