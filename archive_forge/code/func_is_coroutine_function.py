import asyncio
import builtins
import collections
from collections.abc import Generator
import concurrent.futures
import datetime
import functools
from functools import singledispatch
from inspect import isawaitable
import sys
import types
from tornado.concurrent import (
from tornado.ioloop import IOLoop
from tornado.log import app_log
from tornado.util import TimeoutError
import typing
from typing import Union, Any, Callable, List, Type, Tuple, Awaitable, Dict, overload
def is_coroutine_function(func: Any) -> bool:
    """Return whether *func* is a coroutine function, i.e. a function
    wrapped with `~.gen.coroutine`.

    .. versionadded:: 4.5
    """
    return getattr(func, '__tornado_coroutine__', False)