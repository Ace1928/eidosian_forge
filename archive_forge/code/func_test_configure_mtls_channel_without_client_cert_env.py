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
@mock.patch('google.auth.transport._mtls_helper.get_client_cert_and_key', autospec=True)
def test_configure_mtls_channel_without_client_cert_env(self, get_client_cert_and_key):
    callback = mock.Mock()
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock(), http=mock.Mock())
    is_mtls = authed_http.configure_mtls_channel(callback)
    assert not is_mtls
    callback.assert_not_called()
    is_mtls = authed_http.configure_mtls_channel(callback)
    assert not is_mtls
    get_client_cert_and_key.assert_not_called()