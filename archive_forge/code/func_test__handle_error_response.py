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
@pytest.mark.parametrize('retryable', [True, False])
def test__handle_error_response(retryable):
    response_data = {'error': 'help', 'error_description': "I'm alive"}
    with pytest.raises(exceptions.RefreshError) as excinfo:
        _client._handle_error_response(response_data, retryable)
    assert excinfo.value.retryable == retryable
    assert excinfo.match("help: I\\'m alive")