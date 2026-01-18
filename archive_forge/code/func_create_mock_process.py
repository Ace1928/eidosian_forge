import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def create_mock_process(self, output, error):
    mock_process = mock.Mock()
    attrs = {'communicate.return_value': (output, error), 'returncode': 0}
    mock_process.configure_mock(**attrs)
    return mock_process