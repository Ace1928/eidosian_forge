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
def assert_credential_request_kwargs(cls, request_kwargs, headers, url=CREDENTIAL_URL):
    assert request_kwargs['url'] == url
    assert request_kwargs['method'] == 'GET'
    assert request_kwargs['headers'] == headers
    assert request_kwargs.get('body', None) is None