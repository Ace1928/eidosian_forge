import pytest  # type: ignore
from google.auth import exceptions  # type:ignore
def test_default_retryable_exceptions(retryable_exception):
    assert not retryable_exception().retryable