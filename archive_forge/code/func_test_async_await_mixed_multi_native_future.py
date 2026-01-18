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
@gen_test
def test_async_await_mixed_multi_native_future(self):

    @gen.coroutine
    def f1():
        yield gen.moment

    async def f2():
        await f1()
        return 42

    @gen.coroutine
    def f3():
        yield gen.moment
        raise gen.Return(43)
    results = (yield [f2(), f3()])
    self.assertEqual(results, [42, 43])
    self.finished = True