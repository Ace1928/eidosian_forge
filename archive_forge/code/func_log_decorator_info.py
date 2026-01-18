import asyncio
import cProfile
import functools
from functools import wraps
import gc
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from inspect import signature, Parameter
from pathlib import Path
from typing import (
import aiofiles
import msgpack
import zstandard
import pstats
from aiokeydb import AsyncKeyDB, KeyDBClient
from lazyops.utils import logger
from pydantic import BaseModel, ValidationError
import traceback
def log_decorator_info(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to log the entry and exit of an asynchronous function, enhancing the visibility of the function's execution flow.

    This decorator is crucial for tracing the execution flow, especially in asynchronous environments where the concurrent execution of tasks can obscure the order of operations. By logging the start and end of a function's execution, it provides a clear, chronological trace of the function's activity, aiding in debugging and performance monitoring.

    Args:
        func (Callable[..., Awaitable[T]]): The asynchronous function to be decorated, enhancing its logging capabilities.

    Returns:
        Callable[..., Awaitable[T]]: The original function wrapped with logging functionality, preserving its asynchronous nature.

    Example:
        @log_decorator_info
        async def async_function_example(param: int) -> str:
            return f"Processed {param}"
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        func_signature = signature(func)
        bound_arguments = func_signature.bind(*args, **kwargs).arguments
        argument_str = ', '.join([f'{k}={v}' for k, v in bound_arguments.items()])
        logging.info(f'Entering {func.__name__} with arguments: {argument_str}')
        try:
            result: T = await func(*args, **kwargs)
            logging.info(f'Exiting {func.__name__} with result: {result}')
            return result
        except Exception as e:
            logging.error(f'Exception in {func.__name__} with arguments: {argument_str}. Error: {str(e)}', exc_info=True)
            raise
    return wrapper