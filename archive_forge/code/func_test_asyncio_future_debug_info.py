import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
def test_asyncio_future_debug_info(self):
    self.finished = True
    asyncio_loop = asyncio.get_event_loop()
    self.addCleanup(asyncio_loop.set_debug, asyncio_loop.get_debug())
    asyncio_loop.set_debug(True)

    def f():
        yield gen.moment
    coro = gen.coroutine(f)()
    self.assertIsInstance(coro, asyncio.Future)
    expected = 'created at %s:%d' % (__file__, f.__code__.co_firstlineno + 3)
    actual = repr(coro)
    self.assertIn(expected, actual)