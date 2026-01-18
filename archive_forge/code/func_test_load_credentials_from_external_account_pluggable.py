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
def test_load_credentials_from_external_account_pluggable(get_project_id, tmpdir):
    config_file = tmpdir.join('config.json')
    config_file.write(json.dumps(PLUGGABLE_DATA))
    credentials, project_id = _default.load_credentials_from_file(str(config_file))
    assert isinstance(credentials, pluggable.Credentials)
    assert project_id is None
    assert get_project_id.called