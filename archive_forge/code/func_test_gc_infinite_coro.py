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
def test_gc_infinite_coro(self):
    loop = self.get_new_ioloop()
    result = []
    wfut = []

    @gen.coroutine
    def infinite_coro():
        try:
            while True:
                yield gen.sleep(0.001)
                result.append(True)
        finally:
            result.append(None)

    @gen.coroutine
    def do_something():
        fut = infinite_coro()
        fut._refcycle = fut
        wfut.append(weakref.ref(fut))
        yield gen.sleep(0.2)
    loop.run_sync(do_something)
    loop.close()
    gc.collect()
    self.assertIs(wfut[0](), None)
    self.assertGreaterEqual(len(result), 2)
    if not self.is_pypy3():
        self.assertIs(result[-1], None)