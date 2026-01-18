import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
@mock.patch.object(aws.Credentials, '__init__', return_value=None)
def test_from_info_full_options(self, mock_init):
    credentials = aws.Credentials.from_info({'audience': AUDIENCE, 'subject_token_type': SUBJECT_TOKEN_TYPE, 'token_url': TOKEN_URL, 'token_info_url': TOKEN_INFO_URL, 'service_account_impersonation_url': SERVICE_ACCOUNT_IMPERSONATION_URL, 'service_account_impersonation': {'token_lifetime_seconds': 2800}, 'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET, 'quota_project_id': QUOTA_PROJECT_ID, 'credential_source': self.CREDENTIAL_SOURCE})
    assert isinstance(credentials, aws.Credentials)
    mock_init.assert_called_once_with(audience=AUDIENCE, subject_token_type=SUBJECT_TOKEN_TYPE, token_url=TOKEN_URL, token_info_url=TOKEN_INFO_URL, service_account_impersonation_url=SERVICE_ACCOUNT_IMPERSONATION_URL, service_account_impersonation_options={'token_lifetime_seconds': 2800}, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, credential_source=self.CREDENTIAL_SOURCE, quota_project_id=QUOTA_PROJECT_ID, workforce_pool_user_project=None)