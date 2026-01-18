import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
def test_fetch_id_token_credentials_from_metadata_server(monkeypatch):
    monkeypatch.delenv(environment_vars.CREDENTIALS, raising=False)
    mock_req = mock.Mock()
    with mock.patch('google.auth.compute_engine._metadata.ping', return_value=True):
        with mock.patch('google.auth.compute_engine.IDTokenCredentials.__init__', return_value=None) as mock_init:
            id_token.fetch_id_token_credentials(ID_TOKEN_AUDIENCE, request=mock_req)
        mock_init.assert_called_once_with(mock_req, ID_TOKEN_AUDIENCE, use_metadata_identity_endpoint=True)