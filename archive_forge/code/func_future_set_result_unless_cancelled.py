import asyncio
from concurrent import futures
import functools
import sys
import types
from tornado.log import app_log
import typing
from typing import Any, Callable, Optional, Tuple, Union
def future_set_result_unless_cancelled(future: 'Union[futures.Future[_T], Future[_T]]', value: _T) -> None:
    """Set the given ``value`` as the `Future`'s result, if not cancelled.

    Avoids ``asyncio.InvalidStateError`` when calling ``set_result()`` on
    a cancelled `asyncio.Future`.

    .. versionadded:: 5.0
    """
    if not future.cancelled():
        future.set_result(value)