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
@LOAD_FILE_PATCH
@mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', autospec=True)
def test__get_gcloud_sdk_credentials(get_adc_path, load, quota_project_id):
    get_adc_path.return_value = test_default.SERVICE_ACCOUNT_FILE
    credentials, project_id = _default._get_gcloud_sdk_credentials(quota_project_id=quota_project_id)
    assert credentials is MOCK_CREDENTIALS
    assert project_id is mock.sentinel.project_id
    load.assert_called_with(test_default.SERVICE_ACCOUNT_FILE, quota_project_id=quota_project_id)