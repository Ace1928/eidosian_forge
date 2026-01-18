import base64
import datetime
import json
import os
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import jwt
def test__make_jwt_without_audience(self):
    cred = jwt.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO.copy(), subject=self.SUBJECT, audience=None, additional_claims={'scope': 'foo bar'})
    token, _ = cred._make_jwt()
    payload = jwt.decode(token, PUBLIC_CERT_BYTES)
    assert payload['scope'] == 'foo bar'
    assert 'aud' not in payload