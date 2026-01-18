import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.utcfromtimestamp(0))
@mock.patch('google.auth.compute_engine._metadata.get', autospec=True)
@mock.patch('google.auth.iam.Signer.sign', autospec=True)
def test_additional_claims(self, sign, get, utcnow):
    get.side_effect = [{'email': 'service-account@example.com', 'scopes': ['one', 'two']}]
    sign.side_effect = [b'signature']
    request = mock.create_autospec(transport.Request, instance=True)
    self.credentials = credentials.IDTokenCredentials(request=request, target_audience='https://audience.com', additional_claims={'foo': 'bar'})
    token = self.credentials._make_authorization_grant_assertion()
    payload = jwt.decode(token, verify=False)
    assert token.endswith(b'.c2lnbmF0dXJl')
    assert payload == {'aud': 'https://www.googleapis.com/oauth2/v4/token', 'exp': 3600, 'iat': 0, 'iss': 'service-account@example.com', 'target_audience': 'https://audience.com', 'foo': 'bar'}