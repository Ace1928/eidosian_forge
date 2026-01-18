import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _credentials_async as _credentials_async
from google.oauth2 import credentials
from tests.oauth2 import test_credentials
@classmethod
def make_credentials(cls):
    return _credentials_async.Credentials(token=None, refresh_token=cls.REFRESH_TOKEN, token_uri=cls.TOKEN_URI, client_id=cls.CLIENT_ID, client_secret=cls.CLIENT_SECRET, enable_reauth_refresh=True)