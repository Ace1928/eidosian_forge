import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
def test__convert_oauth2_credentials():
    old_credentials = oauth2client.client.OAuth2Credentials('access_token', 'client_id', 'client_secret', 'refresh_token', datetime.datetime.min, 'token_uri', 'user_agent', scopes='one two')
    new_credentials = _oauth2client._convert_oauth2_credentials(old_credentials)
    assert new_credentials.token == old_credentials.access_token
    assert new_credentials._refresh_token == old_credentials.refresh_token
    assert new_credentials._client_id == old_credentials.client_id
    assert new_credentials._client_secret == old_credentials.client_secret
    assert new_credentials._token_uri == old_credentials.token_uri
    assert new_credentials.scopes == old_credentials.scopes