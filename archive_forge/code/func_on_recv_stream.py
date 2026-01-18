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
def on_recv_stream(self, callback: Callable[[ZMQStream, list[zmq.Frame]], Any] | Callable[[ZMQStream, list[bytes]], Any], copy: bool=True):
    """Same as on_recv, but callback will get this stream as first argument

        callback must take exactly two arguments, as it will be called as::

            callback(stream, msg)

        Useful when a single callback should be used with multiple streams.
        """
    if callback is None:
        self.stop_on_recv()
    else:

        def stream_callback(msg):
            return callback(self, msg)
        self.on_recv(stream_callback, copy=copy)