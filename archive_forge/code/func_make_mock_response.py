import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
@classmethod
def make_mock_response(cls, status, data):
    response = mock.create_autospec(transport.Response, instance=True)
    response.status = status
    if isinstance(data, dict):
        response.data = json.dumps(data).encode('utf-8')
    else:
        response.data = data
    return response