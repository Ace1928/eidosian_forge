import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.compute_engine._metadata
from google.oauth2 import _id_token_async as id_token
from google.oauth2 import _service_account_async
from google.oauth2 import id_token as sync_id_token
from tests.oauth2 import test_id_token
def mock_init(self, request, audience, use_metadata_identity_endpoint):
    assert use_metadata_identity_endpoint
    self.token = 'id_token'