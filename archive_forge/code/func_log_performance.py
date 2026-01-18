import collections  # Provides support for ordered dictionaries, which are instrumental in the implementation of the caching mechanism, ensuring FIFO cache eviction logic.
import logging  # Facilitates comprehensive logging capabilities, enabling detailed monitoring and debugging throughout the decorator's operation.
import asyncio  # Essential for the support of asynchronous operations, allowing the decorator to enhance both synchronous and asynchronous functions seamlessly.
import functools  # Offers utilities for working with higher-order functions and operations on callable objects, crucial for the decorator's wrapping mechanism.
import time  # Integral for the execution time measurement and the implementation of retry delays, providing accurate performance metrics and controlled operation retries.
from inspect import (
from typing import (
import tracemalloc  # Activates memory usage tracking, enabling the identification of memory leaks and optimizing the decorator's memory footprint.
def log_performance(self, func: Callable, start_time: float, end_time: float) -> None:
    """
        Logs the performance of the decorated function, adjusting for decorator overhead.

        Args:
            func (F): The function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.
        """
    overhead = 0.0001
    adjusted_time = end_time - start_time - overhead
    logging.debug(f'{func.__name__} executed in {adjusted_time:.6f}s')