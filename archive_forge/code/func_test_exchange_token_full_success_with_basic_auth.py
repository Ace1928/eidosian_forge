import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
def test_exchange_token_full_success_with_basic_auth(self):
    """Test token exchange success with basic client authentication using full
        parameters.
        """
    client = self.make_client(self.CLIENT_AUTH_BASIC)
    headers = self.ADDON_HEADERS.copy()
    headers['Content-Type'] = 'application/x-www-form-urlencoded'
    headers['Authorization'] = 'Basic {}'.format(BASIC_AUTH_ENCODING)
    request_data = {'grant_type': self.GRANT_TYPE, 'resource': self.RESOURCE, 'audience': self.AUDIENCE, 'scope': ' '.join(self.SCOPES), 'requested_token_type': self.REQUESTED_TOKEN_TYPE, 'subject_token': self.SUBJECT_TOKEN, 'subject_token_type': self.SUBJECT_TOKEN_TYPE, 'actor_token': self.ACTOR_TOKEN, 'actor_token_type': self.ACTOR_TOKEN_TYPE, 'options': urllib.parse.quote(json.dumps(self.ADDON_OPTIONS))}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE)
    response = client.exchange_token(request, self.GRANT_TYPE, self.SUBJECT_TOKEN, self.SUBJECT_TOKEN_TYPE, self.RESOURCE, self.AUDIENCE, self.SCOPES, self.REQUESTED_TOKEN_TYPE, self.ACTOR_TOKEN, self.ACTOR_TOKEN_TYPE, self.ADDON_OPTIONS, self.ADDON_HEADERS)
    self.assert_request_kwargs(request.call_args[1], headers, request_data)
    assert response == self.SUCCESS_RESPONSE