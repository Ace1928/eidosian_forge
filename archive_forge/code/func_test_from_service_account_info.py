import datetime
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _service_account_async as service_account
from tests.oauth2 import test_service_account
def test_from_service_account_info(self):
    credentials = service_account.IDTokenCredentials.from_service_account_info(test_service_account.SERVICE_ACCOUNT_INFO, target_audience=self.TARGET_AUDIENCE)
    assert credentials._signer.key_id == test_service_account.SERVICE_ACCOUNT_INFO['private_key_id']
    assert credentials.service_account_email == test_service_account.SERVICE_ACCOUNT_INFO['client_email']
    assert credentials._token_uri == test_service_account.SERVICE_ACCOUNT_INFO['token_uri']
    assert credentials._target_audience == self.TARGET_AUDIENCE