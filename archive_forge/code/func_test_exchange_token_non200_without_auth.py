import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
def test_exchange_token_non200_without_auth(self):
    """Test token exchange without client auth responding with non-200 status.
        """
    client = self.make_client()
    request = self.make_mock_request(status=http_client.BAD_REQUEST, data=self.ERROR_RESPONSE)
    with pytest.raises(exceptions.OAuthError) as excinfo:
        client.exchange_token(request, self.GRANT_TYPE, self.SUBJECT_TOKEN, self.SUBJECT_TOKEN_TYPE, self.RESOURCE, self.AUDIENCE, self.SCOPES, self.REQUESTED_TOKEN_TYPE, self.ACTOR_TOKEN, self.ACTOR_TOKEN_TYPE, self.ADDON_OPTIONS, self.ADDON_HEADERS)
    assert excinfo.match('Error code invalid_request: Invalid subject token - https://tools.ietf.org/html/rfc6749')