import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import transport
from google.auth.impersonated_credentials import Credentials
from google.oauth2 import credentials
from google.oauth2 import service_account
def test_id_token_invalid_cred(self, mock_donor_credentials, mock_authorizedsession_idtoken):
    credentials = None
    with pytest.raises(exceptions.GoogleAuthError) as excinfo:
        impersonated_credentials.IDTokenCredentials(credentials)
    assert excinfo.match('Provided Credential must be impersonated_credentials')