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
@mock.patch('google.auth._credentials_async.with_scopes_if_required', return_value=MOCK_CREDENTIALS, autospec=True)
def test_default_scoped(with_scopes, unused_get):
    scopes = ['one', 'two']
    credentials, project_id = _default.default_async(scopes=scopes)
    assert credentials == with_scopes.return_value
    assert project_id == mock.sentinel.project_id
    with_scopes.assert_called_once_with(MOCK_CREDENTIALS, scopes)