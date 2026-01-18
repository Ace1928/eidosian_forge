import abc
import asyncio
import datetime
import functools
import logging
import os
import random
import threading
import time
from typing import Any, Awaitable, Callable, Generic, Optional, Tuple, Type, TypeVar
from requests import HTTPError
import wandb
from wandb.util import CheckRetryFnType
from .mailbox import ContextCancelledError
def retriable(*args: Any, **kargs: Any) -> Callable[[_F], _F]:

    def decorator(fn: _F) -> _F:
        retrier: Retry[Any] = Retry(fn, *args, **kargs)

        @functools.wraps(fn)
        def wrapped_fn(*args: Any, **kargs: Any) -> Any:
            return retrier(*args, **kargs)
        return wrapped_fn
    return decorator