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
def test_add_future_threads(self):
    with futures.ThreadPoolExecutor(1) as pool:

        def dummy():
            pass
        self.io_loop.add_future(pool.submit(dummy), lambda future: self.stop(future))
        future = self.wait()
        self.assertTrue(future.done())
        self.assertTrue(future.result() is None)