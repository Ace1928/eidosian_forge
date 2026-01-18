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
def test_default_gdch_service_account_credentials(get_adc_path):
    get_adc_path.return_value = GDCH_SERVICE_ACCOUNT_FILE
    creds, project = _default.default(quota_project_id='project-foo')
    assert isinstance(creds, gdch_credentials.ServiceAccountCredentials)
    assert creds._service_identity_name == 'service_identity_name'
    assert creds._audience is None
    assert creds._token_uri == 'https://service-identity.<Domain>/authenticate'
    assert creds._ca_cert_path == '/path/to/ca/cert'
    assert project == 'project_foo'