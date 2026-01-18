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
@pytest.mark.parametrize('can_retry', [True, False])
def test__token_endpoint_request_no_throw_with_retry(can_retry):
    response_data = {'error': 'help', 'error_description': "I'm alive"}
    body = 'dummy body'
    mock_response = mock.create_autospec(transport.Response, instance=True)
    mock_response.status = http_client.INTERNAL_SERVER_ERROR
    mock_response.data = json.dumps(response_data).encode('utf-8')
    mock_request = mock.create_autospec(transport.Request)
    mock_request.return_value = mock_response
    _client._token_endpoint_request_no_throw(mock_request, mock.Mock(), body, mock.Mock(), mock.Mock(), can_retry=can_retry)
    if can_retry:
        assert mock_request.call_count == 4
    else:
        assert mock_request.call_count == 1