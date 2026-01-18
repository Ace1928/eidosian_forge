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
def assert_impersonation_request_kwargs(cls, request_kwargs, headers, request_data, service_account_impersonation_url=SERVICE_ACCOUNT_IMPERSONATION_URL):
    assert request_kwargs['url'] == service_account_impersonation_url
    assert request_kwargs['method'] == 'POST'
    assert request_kwargs['headers'] == headers
    assert request_kwargs['body'] is not None
    body_json = json.loads(request_kwargs['body'].decode('utf-8'))
    assert body_json == request_data