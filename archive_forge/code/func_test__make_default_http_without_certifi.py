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
@mock.patch.object(google.auth.transport.urllib3, 'certifi', new=None)
def test__make_default_http_without_certifi():
    http = google.auth.transport.urllib3._make_default_http()
    assert 'cert_reqs' not in http.connection_pool_kw