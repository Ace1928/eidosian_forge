import logging
import logging.config
import logging.handlers
import sys
import time
import asyncio
import aiofiles
from typing import (
import pathlib
import json
from concurrent.futures import Executor, ThreadPoolExecutor
import functools
from functools import wraps
import tracemalloc
import inspect
from inspect import signature, Parameter
from IndegoValidation import AsyncValidationException, ValidationRules
def dynamic_retry_strategy_sync(self, exception: BaseException, attempt: int) -> int:
    """
        Dynamically determines the retry delay based on the exception type and the number of attempts already made.
        This method allows for a more adaptive and responsive retry mechanism, potentially increasing the chances of success in subsequent attempts.

        Args:
            exception (BaseException): The exception that triggered the retry logic.
            attempt (int): The current retry attempt number.

        Returns:
            int: The delay in seconds before the next retry attempt.
        """
    if isinstance(exception, TimeoutError):
        return min(5, 2 ** attempt)
    elif isinstance(exception, ConnectionError):
        return min(10, 2 * attempt)
    return self.delay