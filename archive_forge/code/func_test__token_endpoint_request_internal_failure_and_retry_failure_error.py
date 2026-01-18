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
def test__token_endpoint_request_internal_failure_and_retry_failure_error():
    retryable_error = mock.create_autospec(transport.Response, instance=True)
    retryable_error.status = http_client.BAD_REQUEST
    retryable_error.data = json.dumps({'error_description': 'internal_failure'}).encode('utf-8')
    unretryable_error = mock.create_autospec(transport.Response, instance=True)
    unretryable_error.status = http_client.BAD_REQUEST
    unretryable_error.data = json.dumps({'error_description': 'invalid_scope'}).encode('utf-8')
    request = mock.create_autospec(transport.Request)
    request.side_effect = [retryable_error, retryable_error, unretryable_error]
    with pytest.raises(exceptions.RefreshError):
        _client._token_endpoint_request(request, 'http://example.com', {'error_description': 'invalid_scope'})
    assert request.call_count == 3