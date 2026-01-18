from __future__ import annotations
import inspect
import logging
from tenacity import retry, wait_exponential, stop_after_delay, before_sleep_log, retry_unless_exception_type, retry_if_exception_type, retry_if_exception
from typing import Optional, Union, Tuple, Type, TYPE_CHECKING
def get_retryable_wrapper(max_attempts: int=15, max_delay: int=60, logging_level: int=logging.DEBUG, **kwargs) -> 'WrappedFn':
    """
    Creates a retryable decorator
    """
    from aiokeydb import exceptions as aiokeydb_exceptions
    from redis import exceptions as redis_exceptions
    return retry(wait=wait_exponential(multiplier=0.5, min=1, max=max_attempts), stop=stop_after_delay(max_delay), before_sleep=before_sleep_log(logger, logging_level), retry=retry_if_type(exception_types=(aiokeydb_exceptions.ConnectionError, aiokeydb_exceptions.TimeoutError, aiokeydb_exceptions.BusyLoadingError, redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, redis_exceptions.BusyLoadingError), excluded_types=(aiokeydb_exceptions.AuthenticationError, aiokeydb_exceptions.AuthorizationError, aiokeydb_exceptions.InvalidResponse, aiokeydb_exceptions.ResponseError, aiokeydb_exceptions.NoScriptError, redis_exceptions.AuthenticationError, redis_exceptions.AuthorizationError, redis_exceptions.InvalidResponse, redis_exceptions.ResponseError, redis_exceptions.NoScriptError)))