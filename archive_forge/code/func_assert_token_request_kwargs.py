import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
@classmethod
def assert_token_request_kwargs(cls, request_kwargs, headers, request_data, token_url=TOKEN_URL):
    assert request_kwargs['url'] == token_url
    assert request_kwargs['method'] == 'POST'
    assert request_kwargs['headers'] == headers
    assert request_kwargs['body'] is not None
    body_tuples = urllib.parse.parse_qsl(request_kwargs['body'])
    assert len(body_tuples) == len(request_data.keys())
    for k, v in body_tuples:
        assert v.decode('utf-8') == request_data[k.decode('utf-8')]