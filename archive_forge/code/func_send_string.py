from __future__ import annotations
import asyncio
import pickle
import warnings
from queue import Queue
from typing import Any, Awaitable, Callable, Sequence, cast, overload
from tornado.ioloop import IOLoop
from tornado.log import gen_log
import zmq
import zmq._future
from zmq import POLLIN, POLLOUT
from zmq._typing import Literal
from zmq.utils import jsonapi
def send_string(self, u: str, flags: int=0, encoding: str='utf-8', callback: Callable | None=None, **kwargs: Any):
    """Send a unicode message with an encoding.
        See zmq.socket.send_unicode for details.
        """
    if not isinstance(u, str):
        raise TypeError('unicode/str objects only')
    return self.send(u.encode(encoding), flags=flags, callback=callback, **kwargs)