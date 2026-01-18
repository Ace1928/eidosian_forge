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
def test_authed_http_defaults(self):
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(mock.sentinel.credentials)
    assert authed_http.credentials == mock.sentinel.credentials
    assert isinstance(authed_http.http, urllib3.PoolManager)