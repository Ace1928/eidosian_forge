import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
def test_info_with_workforce_pool_user_project(self):
    credentials = self.make_credentials(audience=WORKFORCE_AUDIENCE, subject_token_type=WORKFORCE_SUBJECT_TOKEN_TYPE, credential_source=self.CREDENTIAL_SOURCE_TEXT_URL.copy(), workforce_pool_user_project=WORKFORCE_POOL_USER_PROJECT)
    assert credentials.info == {'type': 'external_account', 'audience': WORKFORCE_AUDIENCE, 'subject_token_type': WORKFORCE_SUBJECT_TOKEN_TYPE, 'token_url': TOKEN_URL, 'token_info_url': TOKEN_INFO_URL, 'credential_source': self.CREDENTIAL_SOURCE_TEXT_URL, 'workforce_pool_user_project': WORKFORCE_POOL_USER_PROJECT}