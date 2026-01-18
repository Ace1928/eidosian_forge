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
@EXTERNAL_ACCOUNT_GET_PROJECT_ID_PATCH
def test_default_environ_external_credentials_workforce_impersonated(get_project_id, monkeypatch, tmpdir):
    config_file = tmpdir.join('config.json')
    config_file.write(json.dumps(IMPERSONATED_IDENTITY_POOL_WORKFORCE_DATA))
    monkeypatch.setenv(environment_vars.CREDENTIALS, str(config_file))
    credentials, project_id = _default.default(scopes=['https://www.google.com/calendar/feeds'])
    assert isinstance(credentials, identity_pool.Credentials)
    assert not credentials.is_user
    assert credentials.is_workforce_pool
    assert project_id is mock.sentinel.project_id
    assert credentials.scopes == ['https://www.google.com/calendar/feeds']