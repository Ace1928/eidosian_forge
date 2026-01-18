import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test_apply_client_authentication_options_request_body_no_body(self):
    headers = {'Content-Type': 'application/json'}
    auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_REQUEST_BODY)
    with pytest.raises(exceptions.OAuthError) as excinfo:
        auth_handler.apply_client_authentication_options(headers)
    assert excinfo.match('HTTP request does not support request-body')