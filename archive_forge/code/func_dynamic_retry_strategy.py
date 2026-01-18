import collections  # Provides support for ordered dictionaries, which are instrumental in the implementation of the caching mechanism, ensuring FIFO cache eviction logic.
import logging  # Facilitates comprehensive logging capabilities, enabling detailed monitoring and debugging throughout the decorator's operation.
import asyncio  # Essential for the support of asynchronous operations, allowing the decorator to enhance both synchronous and asynchronous functions seamlessly.
import functools  # Offers utilities for working with higher-order functions and operations on callable objects, crucial for the decorator's wrapping mechanism.
import time  # Integral for the execution time measurement and the implementation of retry delays, providing accurate performance metrics and controlled operation retries.
from inspect import (
from typing import (
import tracemalloc  # Activates memory usage tracking, enabling the identification of memory leaks and optimizing the decorator's memory footprint.
def dynamic_retry_strategy(self, exception: BaseException) -> Tuple[int, int]:
    """
        Determines the retry strategy dynamically based on the exception type.

        Args:
            exception (Exception): The exception that triggered the retry logic.

        Returns:
            Tuple[int, int]: A tuple containing the number of retries and delay in seconds.
        """
    if isinstance(exception, TimeoutError):
        return (5, 1)
    elif isinstance(exception, ConnectionError):
        return (3, 5)
    return (self.retries, self.delay)