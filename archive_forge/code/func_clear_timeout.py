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
def clear_timeout(self) -> None:
    if self.timeout is not None:
        self.io_loop.remove_timeout(self.timeout)