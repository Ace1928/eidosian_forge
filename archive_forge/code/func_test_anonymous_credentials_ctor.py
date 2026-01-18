import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
def test_anonymous_credentials_ctor():
    anon = credentials.AnonymousCredentials()
    assert anon.token is None
    assert anon.expiry is None
    assert not anon.expired
    assert anon.valid