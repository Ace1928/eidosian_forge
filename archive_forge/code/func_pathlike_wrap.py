import os
import asyncio
from types import coroutine
from fileio import PathIO
from io import (
from functools import partial, singledispatch, wraps
from typing import TypeVar, Union
from aiofiles.threadpool.binary import AsyncBufferedIOBase, AsyncBufferedReader, AsyncFileIO
from aiofiles.threadpool.text import AsyncTextIOWrapper
from aiofiles.base import AiofilesContextManager
from tensorflow.python.platform.gfile import GFile
from tensorflow.python.lib.io import file_io as tfio
from tensorflow.python.lib.io import _pywrap_file_io
def pathlike_wrap(func):

    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run