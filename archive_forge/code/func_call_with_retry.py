import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def call_with_retry(f: Callable[[], Any], match: List[str], description: str, *, max_attempts: int=10, max_backoff_s: int=32) -> Any:
    """Retry a function with exponential backoff.

    Args:
        f: The function to retry.
        match: A list of strings to match in the exception message.
        description: An imperitive description of the function being retried. For
            example, "open the file".
        max_attempts: The maximum number of attempts to retry.
        max_backoff_s: The maximum number of seconds to backoff.
    """
    assert max_attempts >= 1, f'`max_attempts` must be positive. Got {max_attempts}.'
    for i in range(max_attempts):
        try:
            return f()
        except Exception as e:
            is_retryable = any([pattern in str(e) for pattern in match])
            if is_retryable and i + 1 < max_attempts:
                backoff = min(2 ** (i + 1) * random.random(), max_backoff_s)
                logger.debug(f'Retrying {i + 1} attempts to {description} after {backoff} seconds.')
                time.sleep(backoff)
            else:
                raise e from None