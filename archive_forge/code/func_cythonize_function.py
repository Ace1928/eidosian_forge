import hashlib
import os
import asyncio
import aiofiles
from inspect import getsource, getmembers, isfunction, isclass, iscoroutinefunction
from Cython.Build import cythonize
from anyio import Path
from setuptools import setup, Extension
import sys
import logging
import logging.config
import pathlib
from typing import (
from indelogging import (
import concurrent_log_handler
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import (
from inspect import iscoroutinefunction
from functools import wraps
@UniversalDecorator()
def cythonize_function(obj: Type) -> Callable[..., Awaitable]:

    async def async_wrapper(*args, **kwargs):
        compiler = CythonCompiler(obj, module_name=obj.__name__, hash_file=FILES['HASH'])
        await compiler.ensure_latest_version_and_execute()
        if iscoroutinefunction(obj):
            return await obj(*args, **kwargs)
        else:
            return obj(*args, **kwargs)

    def sync_wrapper(*args, **kwargs):
        return asyncio.run(async_wrapper(*args, **kwargs))
    if iscoroutinefunction(obj):
        return async_wrapper
    else:
        return sync_wrapper