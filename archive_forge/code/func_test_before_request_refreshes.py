import datetime
import json
import os
import mock
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.oauth2._client.id_token_jwt_grant', autospec=True)
def test_before_request_refreshes(self, id_token_jwt_grant):
    credentials = self.make_credentials()
    token = 'token'
    id_token_jwt_grant.return_value = (token, _helpers.utcnow() + datetime.timedelta(seconds=500), None)
    request = mock.create_autospec(transport.Request, instance=True)
    assert not credentials.valid
    credentials.before_request(request, 'GET', 'http://example.com?a=1#3', {})
    assert id_token_jwt_grant.called
    assert credentials.valid