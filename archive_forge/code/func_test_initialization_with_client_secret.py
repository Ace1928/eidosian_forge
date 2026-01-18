import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test_initialization_with_client_secret(self):
    client_auth = self.make_client_auth(CLIENT_SECRET)
    assert client_auth.client_auth_type == utils.ClientAuthType.basic
    assert client_auth.client_id == CLIENT_ID
    assert client_auth.client_secret == CLIENT_SECRET