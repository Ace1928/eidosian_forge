import asyncio
from concurrent import futures
import functools
import sys
import types
from tornado.log import app_log
import typing
from typing import Any, Callable, Optional, Tuple, Union
def future_set_exception_unless_cancelled(future: 'Union[futures.Future[_T], Future[_T]]', exc: BaseException) -> None:
    """Set the given ``exc`` as the `Future`'s exception.

    If the Future is already canceled, logs the exception instead. If
    this logging is not desired, the caller should explicitly check
    the state of the Future and call ``Future.set_exception`` instead of
    this wrapper.

    Avoids ``asyncio.InvalidStateError`` when calling ``set_exception()`` on
    a cancelled `asyncio.Future`.

    .. versionadded:: 6.0

    """
    if not future.cancelled():
        future.set_exception(exc)
    else:
        app_log.error('Exception after Future was cancelled', exc_info=exc)