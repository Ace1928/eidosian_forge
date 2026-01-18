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
@skipNotCPython
@unittest.skipIf((3,) < sys.version_info < (3, 6), 'asyncio.Future has reference cycles')
def test_coroutine_refcounting(self):

    @gen.coroutine
    def inner():

        class Foo(object):
            pass
        local_var = Foo()
        self.local_ref = weakref.ref(local_var)

        def dummy():
            pass
        yield gen.coroutine(dummy)()
        raise ValueError('Some error')

    @gen.coroutine
    def inner2():
        try:
            yield inner()
        except ValueError:
            pass
    self.io_loop.run_sync(inner2, timeout=3)
    self.assertIs(self.local_ref(), None)
    self.finished = True