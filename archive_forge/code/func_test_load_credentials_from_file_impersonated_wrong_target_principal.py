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
def test_load_credentials_from_file_impersonated_wrong_target_principal(tmpdir):
    with open(IMPERSONATED_SERVICE_ACCOUNT_AUTHORIZED_USER_SOURCE_FILE) as fh:
        impersonated_credentials_info = json.load(fh)
    impersonated_credentials_info['service_account_impersonation_url'] = 'something_wrong'
    jsonfile = tmpdir.join('invalid.json')
    jsonfile.write(json.dumps(impersonated_credentials_info))
    with pytest.raises(exceptions.DefaultCredentialsError) as excinfo:
        _default.load_credentials_from_file(str(jsonfile))
    assert excinfo.match('Cannot extract target principal')