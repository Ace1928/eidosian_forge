import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
def test_fetch_id_token_credentials_invalid_cred_path(monkeypatch):
    not_json_file = os.path.join(os.path.dirname(__file__), '../data/not_exists.json')
    monkeypatch.setenv(environment_vars.CREDENTIALS, not_json_file)
    with pytest.raises(exceptions.DefaultCredentialsError) as excinfo:
        id_token.fetch_id_token_credentials(ID_TOKEN_AUDIENCE)
    assert excinfo.match('GOOGLE_APPLICATION_CREDENTIALS path is either not found or invalid.')