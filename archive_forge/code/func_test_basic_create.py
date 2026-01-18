import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import exceptions
from google.auth import external_account_authorized_user
from google.auth import transport
def test_basic_create(self):
    creds = external_account_authorized_user.Credentials(token=ACCESS_TOKEN, expiry=datetime.datetime.max, scopes=SCOPES, revoke_url=REVOKE_URL)
    assert creds.expiry == datetime.datetime.max
    assert not creds.expired
    assert creds.token == ACCESS_TOKEN
    assert creds.valid
    assert not creds.requires_scopes
    assert creds.scopes == SCOPES
    assert creds.is_user
    assert creds.revoke_url == REVOKE_URL