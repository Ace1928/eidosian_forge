import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
@classmethod
def assert_request_kwargs(cls, request_kwargs, headers, request_data):
    """Asserts the request was called with the expected parameters.
        """
    assert request_kwargs['url'] == cls.TOKEN_EXCHANGE_ENDPOINT
    assert request_kwargs['method'] == 'POST'
    assert request_kwargs['headers'] == headers
    assert request_kwargs['body'] is not None
    body_tuples = urllib.parse.parse_qsl(request_kwargs['body'])
    for k, v in body_tuples:
        assert v.decode('utf-8') == request_data[k.decode('utf-8')]
    assert len(body_tuples) == len(request_data.keys())