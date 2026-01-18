import datetime
import json
import os
import socket
from tempfile import NamedTemporaryFile
import threading
import time
import sys
import google.auth
from google.auth import _helpers
from googleapiclient import discovery
from six.moves import BaseHTTPServer
from google.oauth2 import service_account
import pytest
from mock import patch
def test_configurable_token_lifespan(oidc_credentials, http_request):
    TOKEN_LIFETIME_SECONDS = 2800
    BUFFER_SECONDS = 5

    def check_impersonation_expiration():
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform.read-only'], request=http_request)
        utcmax = _helpers.utcnow() + datetime.timedelta(seconds=TOKEN_LIFETIME_SECONDS)
        utcmin = utcmax - datetime.timedelta(seconds=BUFFER_SECONDS)
        assert utcmin < credentials._impersonated_credentials.expiry <= utcmax
        return True
    with NamedTemporaryFile() as tmpfile:
        tmpfile.write(oidc_credentials.token.encode('utf-8'))
        tmpfile.flush()
        assert get_project_dns(check_impersonation_expiration, {'type': 'external_account', 'audience': _AUDIENCE_OIDC, 'subject_token_type': 'urn:ietf:params:oauth:token-type:jwt', 'token_url': 'https://sts.googleapis.com/v1/token', 'service_account_impersonation_url': 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateAccessToken'.format(oidc_credentials.service_account_email), 'service_account_impersonation': {'token_lifetime_seconds': TOKEN_LIFETIME_SECONDS}, 'credential_source': {'file': tmpfile.name}})