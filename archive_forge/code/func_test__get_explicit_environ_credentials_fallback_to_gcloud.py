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
@pytest.mark.parametrize('quota_project_id', [None, 'project-foo'])
@mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', autospec=True)
@mock.patch('google.auth._default_async._get_gcloud_sdk_credentials', autospec=True)
def test__get_explicit_environ_credentials_fallback_to_gcloud(get_gcloud_creds, get_adc_path, quota_project_id, monkeypatch):
    get_adc_path.return_value = 'filename'
    monkeypatch.setenv(environment_vars.CREDENTIALS, 'filename')
    _default._get_explicit_environ_credentials(quota_project_id=quota_project_id)
    get_gcloud_creds.assert_called_with(quota_project_id=quota_project_id)