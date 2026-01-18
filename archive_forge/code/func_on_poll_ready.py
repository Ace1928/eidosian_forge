from __future__ import annotations
import warnings
from asyncio import Future
from collections import deque
from functools import partial
from itertools import chain
from typing import Any, Awaitable, Callable, NamedTuple, TypeVar, cast, overload
import zmq as _zmq
from zmq import EVENTS, POLLIN, POLLOUT
from zmq._typing import Literal
def on_poll_ready(f):
    if future.done():
        return
    if watcher.cancelled():
        try:
            future.cancel()
        except RuntimeError:
            pass
        return
    if watcher.exception():
        future.set_exception(watcher.exception())
    else:
        try:
            result = super(_AsyncPoller, self).poll(0)
        except Exception as e:
            future.set_exception(e)
        else:
            future.set_result(result)