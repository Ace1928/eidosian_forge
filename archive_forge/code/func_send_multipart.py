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
def send_multipart(self, msg: Sequence[Any], flags: int=0, copy: bool=True, track: bool=False, callback: Callable | None=None, **kwargs: Any) -> None:
    """Send a multipart message, optionally also register a new callback for sends.
        See zmq.socket.send_multipart for details.
        """
    kwargs.update(dict(flags=flags, copy=copy, track=track))
    self._send_queue.put((msg, kwargs))
    callback = callback or self._send_callback
    if callback is not None:
        self.on_send(callback)
    else:
        self.on_send(lambda *args: None)
    self._add_io_state(zmq.POLLOUT)