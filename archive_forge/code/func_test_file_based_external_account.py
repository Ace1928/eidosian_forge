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
def test_file_based_external_account(oidc_credentials, dns_access):
    with NamedTemporaryFile() as tmpfile:
        tmpfile.write(oidc_credentials.token.encode('utf-8'))
        tmpfile.flush()
        assert get_project_dns(dns_access, {'type': 'external_account', 'audience': _AUDIENCE_OIDC, 'subject_token_type': 'urn:ietf:params:oauth:token-type:jwt', 'token_url': 'https://sts.googleapis.com/v1/token', 'service_account_impersonation_url': 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateAccessToken'.format(oidc_credentials.service_account_email), 'credential_source': {'file': tmpfile.name}})