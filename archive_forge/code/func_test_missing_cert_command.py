import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('google.auth.transport._mtls_helper._read_dca_metadata_file', autospec=True)
@mock.patch('google.auth.transport._mtls_helper._check_dca_metadata_path', autospec=True)
def test_missing_cert_command(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file):
    mock_check_dca_metadata_path.return_value = True
    mock_read_dca_metadata_file.return_value = {}
    with pytest.raises(exceptions.ClientCertError):
        _mtls_helper.get_client_ssl_credentials()