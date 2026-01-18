import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
def test_close_file_object(self):
    """When a file object is used instead of a numeric file descriptor,
        the object should be closed (by IOLoop.close(all_fds=True),
        not just the fd.
        """

    class SocketWrapper(object):

        def __init__(self, sockobj):
            self.sockobj = sockobj
            self.closed = False

        def fileno(self):
            return self.sockobj.fileno()

        def close(self):
            self.closed = True
            self.sockobj.close()
    sockobj, port = bind_unused_port()
    socket_wrapper = SocketWrapper(sockobj)
    io_loop = IOLoop()
    io_loop.add_handler(socket_wrapper, lambda fd, events: None, IOLoop.READ)
    io_loop.close(all_fds=True)
    self.assertTrue(socket_wrapper.closed)