import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
@classmethod
def assert_resource_manager_request_kwargs(cls, request_kwargs, project_number, headers):
    assert request_kwargs['url'] == cls.CLOUD_RESOURCE_MANAGER_URL + project_number
    assert request_kwargs['method'] == 'GET'
    assert request_kwargs['headers'] == headers
    assert 'body' not in request_kwargs