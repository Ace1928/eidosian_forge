import pytest  # type: ignore
from google.auth import exceptions  # type:ignore
@pytest.mark.parametrize('retryable', [True, False])
def test_non_retryable_exceptions(non_retryable_exception, retryable):
    non_retryable_exception = non_retryable_exception(retryable=retryable)
    assert not non_retryable_exception.retryable