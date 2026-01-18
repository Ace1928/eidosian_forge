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
def on_send(self, callback: Callable[[Sequence[Any], zmq.MessageTracker | None], Any]):
    """Register a callback to be called on each send

        There will be two arguments::

            callback(msg, status)

        * `msg` will be the list of sendable objects that was just sent
        * `status` will be the return result of socket.send_multipart(msg) -
          MessageTracker or None.

        Non-copying sends return a MessageTracker object whose
        `done` attribute will be True when the send is complete.
        This allows users to track when an object is safe to write to
        again.

        The second argument will always be None if copy=True
        on the send.

        Use on_send_stream(callback) to register a callback that will be passed
        this ZMQStream as the first argument, in addition to the other two.

        on_send(None) disables recv event polling.

        Parameters
        ----------

        callback : callable
            callback must take exactly two arguments, which will be
            the message being sent (always a list),
            and the return result of socket.send_multipart(msg) -
            MessageTracker or None.

            if callback is None, send callbacks are disabled.
        """
    self._check_closed()
    assert callback is None or callable(callback)
    self._send_callback = callback