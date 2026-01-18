import datetime
import json
import os
import mock
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import service_account
def test__with_use_iam_endpoint(self):
    credentials = self.make_credentials()
    new_credentials = credentials._with_use_iam_endpoint(True)
    assert new_credentials._use_iam_endpoint