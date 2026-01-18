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
def test__token_endpoint_request():
    request = make_request({'test': 'response'})
    result = _client._token_endpoint_request(request, 'http://example.com', {'test': 'params'})
    request.assert_called_with(method='POST', url='http://example.com', headers={'Content-Type': 'application/x-www-form-urlencoded'}, body='test=params'.encode('utf-8'))
    assert result == {'test': 'response'}