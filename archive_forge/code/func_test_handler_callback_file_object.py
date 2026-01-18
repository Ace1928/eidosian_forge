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
def test_handler_callback_file_object(self):
    """The handler callback receives the same fd object it passed in."""
    server_sock, port = bind_unused_port()
    fds = []

    def handle_connection(fd, events):
        fds.append(fd)
        conn, addr = server_sock.accept()
        conn.close()
        self.stop()
    self.io_loop.add_handler(server_sock, handle_connection, IOLoop.READ)
    with contextlib.closing(socket.socket()) as client_sock:
        client_sock.connect(('127.0.0.1', port))
        self.wait()
    self.io_loop.remove_handler(server_sock)
    self.io_loop.add_handler(server_sock.fileno(), handle_connection, IOLoop.READ)
    with contextlib.closing(socket.socket()) as client_sock:
        client_sock.connect(('127.0.0.1', port))
        self.wait()
    self.assertIs(fds[0], server_sock)
    self.assertEqual(fds[1], server_sock.fileno())
    self.io_loop.remove_handler(server_sock.fileno())
    server_sock.close()