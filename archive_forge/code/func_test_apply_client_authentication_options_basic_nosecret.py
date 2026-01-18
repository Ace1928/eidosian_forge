import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test_apply_client_authentication_options_basic_nosecret(self):
    headers = {'Content-Type': 'application/json'}
    request_body = {'foo': 'bar'}
    auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_BASIC_SECRETLESS)
    auth_handler.apply_client_authentication_options(headers, request_body)
    assert headers == {'Content-Type': 'application/json', 'Authorization': 'Basic {}'.format(BASIC_AUTH_ENCODING_SECRETLESS)}
    assert request_body == {'foo': 'bar'}