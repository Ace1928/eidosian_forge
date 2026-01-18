import asyncio
import logging
import logging.config
from typing import List, Tuple, Optional, Union, Any
import functools
from functools import wraps
import cProfile
import pstats
import io
import tracemalloc
import signal
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from memory_profiler import profile
import cachetools.func
from cachetools import TTLCache
import aiofiles
import aiohttp
from aiohttp import web
import json
from datetime import datetime
from sympy import true
def async_wrap(func):

    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = functools.partial(func, *args, **kwargs)
        async with profiled():
            result = await loop.run_in_executor(executor, pfunc)
            logging.debug(f'Function {func.__name__} executed asynchronously with result: {result}')
            return result
    return run