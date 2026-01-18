import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import credentials
def test_invalid_refresh_handler(self):
    scopes = ['email', 'profile']
    with pytest.raises(TypeError) as excinfo:
        credentials.Credentials(token=None, refresh_token=None, token_uri=None, client_id=None, client_secret=None, rapt_token=None, scopes=scopes, default_scopes=None, refresh_handler=object())
    assert excinfo.match('The provided refresh_handler is not a callable or None.')