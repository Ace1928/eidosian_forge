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
def test_bogus_yield_tuple(self):

    @gen.coroutine
    def f():
        yield (1, 2)
    self.assertRaises(gen.BadYieldError, self.io_loop.run_sync, f)