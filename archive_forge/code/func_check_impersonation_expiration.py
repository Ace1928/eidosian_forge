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
def check_impersonation_expiration():
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform.read-only'], request=http_request)
    utcmax = _helpers.utcnow() + datetime.timedelta(seconds=TOKEN_LIFETIME_SECONDS)
    utcmin = utcmax - datetime.timedelta(seconds=BUFFER_SECONDS)
    assert utcmin < credentials._impersonated_credentials.expiry <= utcmax
    return True