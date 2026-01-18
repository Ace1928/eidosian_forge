import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
def test_anonymous_credentials_before_request():
    anon = credentials.AnonymousCredentials()
    request = object()
    method = 'GET'
    url = 'https://example.com/api/endpoint'
    headers = {}
    anon.before_request(request, method, url, headers)
    assert headers == {}