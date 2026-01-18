import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
def test_create_scoped_if_required_scoped():
    unscoped_credentials = RequiresScopedCredentialsImpl()
    scoped_credentials = credentials.with_scopes_if_required(unscoped_credentials, ['one', 'two'])
    assert scoped_credentials is not unscoped_credentials
    assert not scoped_credentials.requires_scopes
    assert scoped_credentials.has_scopes(['one', 'two'])