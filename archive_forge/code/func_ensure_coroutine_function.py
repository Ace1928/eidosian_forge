from __future__ import annotations
import os
import time
import uuid
import random
import typing
import logging
import traceback
import contextlib
import contextvars
import asyncio
import functools
import inspect
from concurrent import futures
from lazyops.utils.helpers import import_function, timer, build_batches
from lazyops.utils.ahelpers import amap_v2 as concurrent_map
def ensure_coroutine_function(func):
    if asyncio.iscoroutinefunction(func):
        return func

    async def wrapped(*args, **kwargs):
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(executor=None, func=lambda: ctx.run(func, *args, **kwargs))
    return wrapped