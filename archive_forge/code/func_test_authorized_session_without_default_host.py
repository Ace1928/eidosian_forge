import datetime
import functools
import os
import sys
import freezegun
import mock
import OpenSSL
import pytest  # type: ignore
import requests
import requests.adapters
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._custom_tls_signer
import google.auth.transport._mtls_helper
import google.auth.transport.requests
from google.oauth2 import service_account
from tests.transport import compliance
def test_authorized_session_without_default_host(self):
    credentials = mock.create_autospec(service_account.Credentials)
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
    authed_session.credentials._create_self_signed_jwt.assert_called_once_with(None)