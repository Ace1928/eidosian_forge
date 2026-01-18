import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
def test_info_with_credential_source(self):
    credentials = self.make_pluggable(credential_source=self.CREDENTIAL_SOURCE.copy())
    assert credentials.info == {'type': 'external_account', 'audience': AUDIENCE, 'subject_token_type': SUBJECT_TOKEN_TYPE, 'token_url': TOKEN_URL, 'token_info_url': TOKEN_INFO_URL, 'credential_source': self.CREDENTIAL_SOURCE}