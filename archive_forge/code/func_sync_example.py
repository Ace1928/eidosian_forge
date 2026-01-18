import collections  # For ordered dictionary support in caching mechanism
import logging  # For logging support
import asyncio  # For handling asynchronous operations
import functools  # For higher-order functions and operations on callable objects
import time  # For measuring execution time and implementing delays
from inspect import (
from typing import (
import tracemalloc  # For tracking memory usage and identifying memory leaks
@StandardDecorator(retries=2, delay=1, log_level=logging.INFO, validation_rules={'x': lambda x: x > 0})
def sync_example(x: int) -> int:
    """Synchronous test function that raises a ValueError for specific input to test retries."""
    if x == 5:
        raise ValueError('Example of a retry scenario.')
    return x * 2