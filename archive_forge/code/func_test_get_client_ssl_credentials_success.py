import datetime
import os
import time
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
def test_get_client_ssl_credentials_success(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_get_client_ssl_credentials, mock_ssl_channel_credentials):
    mock_check_dca_metadata_path.return_value = METADATA_PATH
    mock_read_dca_metadata_file.return_value = {'cert_provider_command': ['some command']}
    mock_get_client_ssl_credentials.return_value = (True, PUBLIC_CERT_BYTES, PRIVATE_KEY_BYTES, None)
    with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
        ssl_credentials = google.auth.transport.grpc.SslCredentials()
    assert ssl_credentials.ssl_credentials is not None
    assert ssl_credentials.is_mtls
    mock_get_client_ssl_credentials.assert_called_once()
    mock_ssl_channel_credentials.assert_called_once_with(certificate_chain=PUBLIC_CERT_BYTES, private_key=PRIVATE_KEY_BYTES)