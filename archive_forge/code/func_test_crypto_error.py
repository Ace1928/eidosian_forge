import os
import sys
import mock
import OpenSSL
import pytest  # type: ignore
from six.moves import http_client
import urllib3  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._mtls_helper
import google.auth.transport.urllib3
from google.oauth2 import service_account
from tests.transport import compliance
def test_crypto_error(self):
    with pytest.raises(OpenSSL.crypto.Error):
        google.auth.transport.urllib3._make_mutual_tls_http(b'invalid cert', b'invalid key')