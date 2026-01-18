import json
import os
import mock
import pytest  # type: ignore
from google.auth import _default
from google.auth import api_key
from google.auth import app_engine
from google.auth import aws
from google.auth import compute_engine
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import impersonated_credentials
from google.auth import pluggable
from google.oauth2 import gdch_credentials
from google.oauth2 import service_account
import google.oauth2.credentials
@mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', autospec=True)
def test_default_impersonated_service_account(get_adc_path):
    get_adc_path.return_value = IMPERSONATED_SERVICE_ACCOUNT_AUTHORIZED_USER_SOURCE_FILE
    credentials, _ = _default.default()
    assert isinstance(credentials, impersonated_credentials.Credentials)
    assert isinstance(credentials._source_credentials, google.oauth2.credentials.Credentials)
    assert credentials.service_account_email == 'service-account-target@example.com'
    assert credentials._delegates == ['service-account-delegate@example.com']
    assert not credentials._quota_project_id
    assert not credentials._target_scopes