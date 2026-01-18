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
def handle_yield(self, yielded: _Yieldable) -> bool:
    try:
        self.future = convert_yielded(yielded)
    except BadYieldError:
        self.future = Future()
        future_set_exc_info(self.future, sys.exc_info())
    if self.future is moment:
        self.io_loop.add_callback(self.ctx_run, self.run)
        return False
    elif self.future is None:
        raise Exception('no pending future')
    elif not self.future.done():

        def inner(f: Any) -> None:
            f = None
            self.ctx_run(self.run)
        self.io_loop.add_future(self.future, inner)
        return False
    return True