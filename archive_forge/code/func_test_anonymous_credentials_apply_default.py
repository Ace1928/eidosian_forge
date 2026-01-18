import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
def test_anonymous_credentials_apply_default():
    anon = credentials.AnonymousCredentials()
    headers = {}
    anon.apply(headers)
    assert headers == {}
    with pytest.raises(ValueError):
        anon.apply(headers, token='TOKEN')