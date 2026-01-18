import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_credential_source_timeout_missing_will_use_default_timeout_value(self):
    CREDENTIAL_SOURCE = {'executable': {'command': self.CREDENTIAL_SOURCE_EXECUTABLE_COMMAND, 'output_file': self.CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}}
    credentials = self.make_pluggable(credential_source=CREDENTIAL_SOURCE)
    assert credentials._credential_source_executable_timeout_millis == pluggable.EXECUTABLE_TIMEOUT_MILLIS_DEFAULT