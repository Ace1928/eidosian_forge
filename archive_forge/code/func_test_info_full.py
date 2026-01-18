import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import exceptions
from google.auth import external_account_authorized_user
from google.auth import transport
def test_info_full(self):
    creds = self.make_credentials(token=ACCESS_TOKEN, expiry=NOW, revoke_url=REVOKE_URL, quota_project_id=QUOTA_PROJECT_ID)
    info = creds.info
    assert info['audience'] == AUDIENCE
    assert info['refresh_token'] == REFRESH_TOKEN
    assert info['token_url'] == TOKEN_URL
    assert info['token_info_url'] == TOKEN_INFO_URL
    assert info['client_id'] == CLIENT_ID
    assert info['client_secret'] == CLIENT_SECRET
    assert info['token'] == ACCESS_TOKEN
    assert info['expiry'] == NOW.isoformat() + 'Z'
    assert info['revoke_url'] == REVOKE_URL
    assert info['quota_project_id'] == QUOTA_PROJECT_ID