from __future__ import annotations
import inspect
import logging
from tenacity import retry, wait_exponential, stop_after_delay, before_sleep_log, retry_unless_exception_type, retry_if_exception_type, retry_if_exception
from typing import Optional, Union, Tuple, Type, TYPE_CHECKING
def create_retryable_client(client: Type[Union[KeyDB, AsyncKeyDB]], max_attempts: int=15, max_delay: int=60, logging_level: int=logging.DEBUG, verbose: Optional[bool]=False, **kwargs) -> Type[Union[KeyDB, AsyncKeyDB]]:
    """
    Creates a retryable client
    """
    if hasattr(client, '_is_retryable_wrapped'):
        return client
    decorator = get_retryable_wrapper(max_attempts=max_attempts, max_delay=max_delay, logging_level=logging_level, **kwargs)
    for attr in dir(client):
        if attr.startswith('_'):
            continue
        if attr in _excluded_funcs:
            continue
        attr_val = getattr(client, attr)
        if inspect.isfunction(attr_val) or inspect.iscoroutinefunction(attr_val):
            if verbose:
                logger.info(f'Wrapping {attr} with retryable decorator')
            setattr(client, attr, decorator(attr_val))
    setattr(client, '_is_retryable_wrapped', True)
    return client