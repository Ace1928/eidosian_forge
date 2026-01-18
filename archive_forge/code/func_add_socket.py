import errno
import os
import socket
import ssl
from tornado import gen
from tornado.log import app_log
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream, SSLIOStream
from tornado.netutil import (
from tornado import process
from tornado.util import errno_from_exception
import typing
from typing import Union, Dict, Any, Iterable, Optional, Awaitable
def add_socket(self, socket: socket.socket) -> None:
    """Singular version of `add_sockets`.  Takes a single socket object."""
    self.add_sockets([socket])