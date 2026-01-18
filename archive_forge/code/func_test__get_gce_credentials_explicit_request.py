import json
import os
import mock
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _default_async as _default
from google.auth import app_engine
from google.auth import compute_engine
from google.auth import environment_vars
from google.auth import exceptions
from google.oauth2 import _service_account_async as service_account
import google.oauth2.credentials
from tests import test__default as test_default
@mock.patch('google.auth.compute_engine._metadata.ping', return_value=False, autospec=True)
def test__get_gce_credentials_explicit_request(ping):
    _default._get_gce_credentials(mock.sentinel.request)
    ping.assert_called_with(request=mock.sentinel.request)