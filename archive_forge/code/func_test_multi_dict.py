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
def test_multi_dict(self):

    @gen.coroutine
    def f():
        results = (yield dict(foo=self.add_one_async(1), bar=self.add_one_async(2)))
        self.assertEqual(results, dict(foo=2, bar=3))
    self.io_loop.run_sync(f)