import datetime
import json
import os
import mock
import pytest  # type: ignore
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _client
def test__can_retry_retryable():
    retryable_codes = transport.DEFAULT_RETRYABLE_STATUS_CODES
    for status_code in range(100, 600):
        if status_code in retryable_codes:
            assert _client._can_retry(status_code, {'error': 'invalid_scope'})
        else:
            assert not _client._can_retry(status_code, {'error': 'invalid_scope'})