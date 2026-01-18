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
def send_pyobj(self, obj: Any, flags: int=0, protocol: int=-1, callback: Callable | None=None, **kwargs: Any):
    """Send a Python object as a message using pickle to serialize.

        See zmq.socket.send_json for details.
        """
    msg = pickle.dumps(obj, protocol)
    return self.send(msg, flags, callback=callback, **kwargs)