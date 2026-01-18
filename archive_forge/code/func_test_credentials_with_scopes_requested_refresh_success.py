import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import credentials
@mock.patch('google.oauth2.reauth.refresh_grant', autospec=True)
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min + _helpers.REFRESH_THRESHOLD)
def test_credentials_with_scopes_requested_refresh_success(self, unused_utcnow, refresh_grant):
    scopes = ['email', 'profile']
    default_scopes = ['https://www.googleapis.com/auth/cloud-platform']
    token = 'token'
    new_rapt_token = 'new_rapt_token'
    expiry = _helpers.utcnow() + datetime.timedelta(seconds=500)
    grant_response = {'id_token': mock.sentinel.id_token, 'scope': 'email profile'}
    refresh_grant.return_value = (token, None, expiry, grant_response, new_rapt_token)
    request = mock.create_autospec(transport.Request)
    creds = credentials.Credentials(token=None, refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, scopes=scopes, default_scopes=default_scopes, rapt_token=self.RAPT_TOKEN, enable_reauth_refresh=True)
    creds.refresh(request)
    refresh_grant.assert_called_with(request, self.TOKEN_URI, self.REFRESH_TOKEN, self.CLIENT_ID, self.CLIENT_SECRET, scopes, self.RAPT_TOKEN, True)
    assert creds.token == token
    assert creds.expiry == expiry
    assert creds.id_token == mock.sentinel.id_token
    assert creds.has_scopes(scopes)
    assert creds.rapt_token == new_rapt_token
    assert creds.granted_scopes == scopes
    assert creds.valid