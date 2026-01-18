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
@mock.patch('google.auth._cloud_sdk.get_project_id', return_value=mock.sentinel.project_id, autospec=True)
@mock.patch('os.path.isfile', return_value=True, autospec=True)
@LOAD_FILE_PATCH
def test__get_gcloud_sdk_credentials_project_id(load, unused_isfile, get_project_id):
    load.return_value = (MOCK_CREDENTIALS, None)
    credentials, project_id = _default._get_gcloud_sdk_credentials()
    assert credentials == MOCK_CREDENTIALS
    assert project_id == mock.sentinel.project_id
    assert get_project_id.called