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
def multi_future(children: Union[List[_Yieldable], Dict[Any, _Yieldable]], quiet_exceptions: 'Union[Type[Exception], Tuple[Type[Exception], ...]]'=()) -> 'Union[Future[List], Future[Dict]]':
    """Wait for multiple asynchronous futures in parallel.

    Since Tornado 6.0, this function is exactly the same as `multi`.

    .. versionadded:: 4.0

    .. versionchanged:: 4.2
       If multiple ``Futures`` fail, any exceptions after the first (which is
       raised) will be logged. Added the ``quiet_exceptions``
       argument to suppress this logging for selected exception types.

    .. deprecated:: 4.3
       Use `multi` instead.
    """
    if isinstance(children, dict):
        keys = list(children.keys())
        children_seq = children.values()
    else:
        keys = None
        children_seq = children
    children_futs = list(map(convert_yielded, children_seq))
    assert all((is_future(i) or isinstance(i, _NullFuture) for i in children_futs))
    unfinished_children = set(children_futs)
    future = _create_future()
    if not children_futs:
        future_set_result_unless_cancelled(future, {} if keys is not None else [])

    def callback(fut: Future) -> None:
        unfinished_children.remove(fut)
        if not unfinished_children:
            result_list = []
            for f in children_futs:
                try:
                    result_list.append(f.result())
                except Exception as e:
                    if future.done():
                        if not isinstance(e, quiet_exceptions):
                            app_log.error('Multiple exceptions in yield list', exc_info=True)
                    else:
                        future_set_exc_info(future, sys.exc_info())
            if not future.done():
                if keys is not None:
                    future_set_result_unless_cancelled(future, dict(zip(keys, result_list)))
                else:
                    future_set_result_unless_cancelled(future, result_list)
    listening = set()
    for f in children_futs:
        if f not in listening:
            listening.add(f)
            future_add_done_callback(f, callback)
    return future