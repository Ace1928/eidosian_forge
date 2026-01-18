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
@pytest.fixture
def mock_donor_credentials():
    with mock.patch('google.oauth2._client.jwt_grant', autospec=True) as grant:
        grant.return_value = ('source token', _helpers.utcnow() + datetime.timedelta(seconds=500), {})
        yield grant