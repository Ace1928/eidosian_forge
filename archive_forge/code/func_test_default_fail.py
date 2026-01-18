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
@mock.patch('google.auth._default_async._get_explicit_environ_credentials', return_value=(None, None), autospec=True)
@mock.patch('google.auth._default_async._get_gcloud_sdk_credentials', return_value=(None, None), autospec=True)
@mock.patch('google.auth._default_async._get_gae_credentials', return_value=(None, None), autospec=True)
@mock.patch('google.auth._default_async._get_gce_credentials', return_value=(None, None), autospec=True)
def test_default_fail(unused_gce, unused_gae, unused_sdk, unused_explicit):
    with pytest.raises(exceptions.DefaultCredentialsError):
        assert _default.default_async()