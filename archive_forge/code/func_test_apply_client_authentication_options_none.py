import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test_apply_client_authentication_options_none(self):
    headers = {'Content-Type': 'application/json'}
    request_body = {'foo': 'bar'}
    auth_handler = self.make_oauth_client_auth_handler()
    auth_handler.apply_client_authentication_options(headers, request_body)
    assert headers == {'Content-Type': 'application/json'}
    assert request_body == {'foo': 'bar'}