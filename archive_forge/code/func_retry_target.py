from __future__ import annotations
import functools
import sys
import time
import inspect
import warnings
from typing import Any, Callable, Iterable, TypeVar, TYPE_CHECKING
from google.api_core.retry.retry_base import _BaseRetry
from google.api_core.retry.retry_base import _retry_error_helper
from google.api_core.retry.retry_base import exponential_sleep_generator
from google.api_core.retry.retry_base import build_retry_error
from google.api_core.retry.retry_base import RetryFailureReason
def retry_target(target: Callable[_P, _R], predicate: Callable[[Exception], bool], sleep_generator: Iterable[float], timeout: float | None=None, on_error: Callable[[Exception], None] | None=None, exception_factory: Callable[[list[Exception], RetryFailureReason, float | None], tuple[Exception, Exception | None]]=build_retry_error, **kwargs):
    """Call a function and retry if it fails.

    This is the lowest-level retry helper. Generally, you'll use the
    higher-level retry helper :class:`Retry`.

    Args:
        target(Callable): The function to call and retry. This must be a
            nullary function - apply arguments with `functools.partial`.
        predicate (Callable[Exception]): A callable used to determine if an
            exception raised by the target should be considered retryable.
            It should return True to retry or False otherwise.
        sleep_generator (Iterable[float]): An infinite iterator that determines
            how long to sleep between retries.
        timeout (Optional[float]): How long to keep retrying the target.
            Note: timeout is only checked before initiating a retry, so the target may
            run past the timeout value as long as it is healthy.
        on_error (Optional[Callable[Exception]]): If given, the on_error
            callback will be called with each retryable exception raised by the
            target. Any error raised by this function will *not* be caught.
        exception_factory: A function that is called when the retryable reaches
            a terminal failure state, used to construct an exception to be raised.
            It takes a list of all exceptions encountered, a retry.RetryFailureReason
            enum indicating the failure cause, and the original timeout value
            as arguments. It should return a tuple of the exception to be raised,
            along with the cause exception if any. The default implementation will raise
            a RetryError on timeout, or the last exception encountered otherwise.
        deadline (float): DEPRECATED: use ``timeout`` instead. For backward
            compatibility, if specified it will override ``timeout`` parameter.

    Returns:
        Any: the return value of the target function.

    Raises:
        ValueError: If the sleep generator stops yielding values.
        Exception: a custom exception specified by the exception_factory if provided.
            If no exception_factory is provided:
                google.api_core.RetryError: If the timeout is exceeded while retrying.
                Exception: If the target raises an error that isn't retryable.
    """
    timeout = kwargs.get('deadline', timeout)
    deadline = time.monotonic() + timeout if timeout is not None else None
    error_list: list[Exception] = []
    for sleep in sleep_generator:
        try:
            result = target()
            if inspect.isawaitable(result):
                warnings.warn(_ASYNC_RETRY_WARNING)
            return result
        except Exception as exc:
            _retry_error_helper(exc, deadline, sleep, error_list, predicate, on_error, exception_factory, timeout)
            time.sleep(sleep)
    raise ValueError('Sleep generator stopped yielding sleep values.')