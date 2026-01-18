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
def test_apply_with_quota_project_id(self):
    creds = _credentials_async.Credentials(token='token', refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, quota_project_id='quota-project-123')
    headers = {}
    creds.apply(headers)
    assert headers['x-goog-user-project'] == 'quota-project-123'