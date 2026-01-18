from __future__ import annotations
import time
import uuid
import typing
import random
import inspect
import functools
import datetime
import itertools
import asyncio
import contextlib
import async_lru
import signal
from pathlib import Path
from frozendict import frozendict
from typing import Dict, Callable, List, Any, Union, Coroutine, TypeVar, Optional, TYPE_CHECKING
from lazyops.utils.logs import default_logger
from lazyops.utils.serialization import (
from lazyops.utils.lazy import (
def create_background_task(func: Union[Callable, Coroutine], *args, **kwargs):
    """
    Creates a background task and adds it to the global set of background tasks
    """
    if inspect.isawaitable(func):
        task = asyncio.create_task(func)
    else:
        task = asyncio.create_task(run_as_coro(func, *args, **kwargs))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task