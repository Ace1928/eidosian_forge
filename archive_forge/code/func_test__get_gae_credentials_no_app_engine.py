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
@mock.patch.dict(os.environ)
def test__get_gae_credentials_no_app_engine():
    assert environment_vars.LEGACY_APPENGINE_RUNTIME not in os.environ
    import sys
    with mock.patch.dict(sys.modules, {'google.auth.app_engine': None}):
        credentials, project_id = _default._get_gae_credentials()
        assert credentials is None
        assert project_id is None
        os.environ[environment_vars.LEGACY_APPENGINE_RUNTIME] = 'python27'
        credentials, project_id = _default._get_gae_credentials()
        assert credentials is None
        assert project_id is None