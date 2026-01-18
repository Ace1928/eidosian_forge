import datetime
import json
import os
import mock
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.auth.jwt.Credentials', instance=True, autospec=True)
def test__create_self_signed_jwt_always_use_jwt_access_with_audience_similar_jwt_is_reused(self, jwt):
    credentials = service_account.Credentials(SIGNER, self.SERVICE_ACCOUNT_EMAIL, self.TOKEN_URI, default_scopes=['bar', 'foo'], always_use_jwt_access=True)
    audience = 'https://pubsub.googleapis.com'
    credentials._create_self_signed_jwt(audience)
    credentials._jwt_credentials._audience = audience
    credentials._create_self_signed_jwt(audience)
    jwt.from_signing_credentials.assert_called_once_with(credentials, audience)