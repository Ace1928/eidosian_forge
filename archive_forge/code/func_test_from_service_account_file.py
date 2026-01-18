import datetime
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _service_account_async as service_account
from tests.oauth2 import test_service_account
def test_from_service_account_file(self):
    info = test_service_account.SERVICE_ACCOUNT_INFO.copy()
    credentials = service_account.IDTokenCredentials.from_service_account_file(test_service_account.SERVICE_ACCOUNT_JSON_FILE, target_audience=self.TARGET_AUDIENCE)
    assert credentials.service_account_email == info['client_email']
    assert credentials._signer.key_id == info['private_key_id']
    assert credentials._token_uri == info['token_uri']
    assert credentials._target_audience == self.TARGET_AUDIENCE