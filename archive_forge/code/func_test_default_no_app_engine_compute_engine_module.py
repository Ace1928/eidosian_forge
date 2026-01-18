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
@mock.patch('google.auth._default_async._get_explicit_environ_credentials', return_value=(MOCK_CREDENTIALS, mock.sentinel.project_id), autospec=True)
def test_default_no_app_engine_compute_engine_module(unused_get):
    """
    google.auth.compute_engine and google.auth.app_engine are both optional
    to allow not including them when using this package. This verifies
    that default fails gracefully if these modules are absent
    """
    import sys
    with mock.patch.dict('sys.modules'):
        sys.modules['google.auth.compute_engine'] = None
        sys.modules['google.auth.app_engine'] = None
        assert _default.default_async() == (MOCK_CREDENTIALS, mock.sentinel.project_id)