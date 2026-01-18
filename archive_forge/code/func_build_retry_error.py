from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def build_retry_error(exc_list: list[Exception], reason: RetryFailureReason, timeout_val: float | None, **kwargs: Any) -> tuple[Exception, Exception | None]:
    """
    Default exception_factory implementation.

    Returns a RetryError if the failure is due to a timeout, otherwise
    returns the last exception encountered.

    Args:
      - exc_list: list of exceptions that occurred during the retry
      - reason: reason for the retry failure.
            Can be TIMEOUT or NON_RETRYABLE_ERROR
      - timeout_val: the original timeout value for the retry (in seconds), for use in the exception message

    Returns:
      - tuple: a tuple of the exception to be raised, and the cause exception if any
    """
    if reason == RetryFailureReason.TIMEOUT:
        src_exc = exc_list[-1] if exc_list else None
        timeout_val_str = f'of {timeout_val:0.1f}s ' if timeout_val is not None else ''
        return (exceptions.RetryError(f'Timeout {timeout_val_str}exceeded', src_exc), src_exc)
    elif exc_list:
        return (exc_list[-1], None)
    else:
        return (exceptions.RetryError('Unknown error', None), None)