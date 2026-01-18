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
@pytest.mark.parametrize('response_data', [{'error': 'internal_failure'}, {'error': 'server_error'}])
def test__can_retry_message(response_data):
    assert _client._can_retry(http_client.OK, response_data)