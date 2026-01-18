import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
def test_info_with_full_options(self):
    credentials = self.make_credentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, quota_project_id=self.QUOTA_PROJECT_ID, token_info_url=self.TOKEN_INFO_URL, service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL, service_account_impersonation_options={'token_lifetime_seconds': 2800})
    assert credentials.info == {'type': 'external_account', 'audience': self.AUDIENCE, 'subject_token_type': self.SUBJECT_TOKEN_TYPE, 'token_url': self.TOKEN_URL, 'token_info_url': self.TOKEN_INFO_URL, 'service_account_impersonation_url': self.SERVICE_ACCOUNT_IMPERSONATION_URL, 'service_account_impersonation': {'token_lifetime_seconds': 2800}, 'credential_source': self.CREDENTIAL_SOURCE.copy(), 'quota_project_id': self.QUOTA_PROJECT_ID, 'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}