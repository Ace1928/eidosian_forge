import functools
import socket
import numbers
import datetime
import ssl
import typing
from tornado.concurrent import Future, future_add_done_callback
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream
from tornado import gen
from tornado.netutil import Resolver
from tornado.gen import TimeoutError
from typing import Any, Union, Dict, Tuple, List, Callable, Iterator, Optional
def on_connect_done(self, addrs: Iterator[Tuple[socket.AddressFamily, Tuple]], af: socket.AddressFamily, addr: Tuple, future: 'Future[IOStream]') -> None:
    self.remaining -= 1
    try:
        stream = future.result()
    except Exception as e:
        if self.future.done():
            return
        self.last_error = e
        self.try_connect(addrs)
        if self.timeout is not None:
            self.io_loop.remove_timeout(self.timeout)
            self.on_timeout()
        return
    self.clear_timeouts()
    if self.future.done():
        stream.close()
    else:
        self.streams.discard(stream)
        self.future.set_result((af, addr, stream))
        self.close_streams()