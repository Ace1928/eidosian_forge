import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
@mock.patch.object(identity_pool.Credentials, '__init__', return_value=None)
def test_from_info_workforce_pool(self, mock_init):
    credentials = identity_pool.Credentials.from_info({'audience': WORKFORCE_AUDIENCE, 'subject_token_type': WORKFORCE_SUBJECT_TOKEN_TYPE, 'token_url': TOKEN_URL, 'credential_source': self.CREDENTIAL_SOURCE_TEXT, 'workforce_pool_user_project': WORKFORCE_POOL_USER_PROJECT})
    assert isinstance(credentials, identity_pool.Credentials)
    mock_init.assert_called_once_with(audience=WORKFORCE_AUDIENCE, subject_token_type=WORKFORCE_SUBJECT_TOKEN_TYPE, token_url=TOKEN_URL, token_info_url=None, service_account_impersonation_url=None, service_account_impersonation_options={}, client_id=None, client_secret=None, credential_source=self.CREDENTIAL_SOURCE_TEXT, quota_project_id=None, workforce_pool_user_project=WORKFORCE_POOL_USER_PROJECT)