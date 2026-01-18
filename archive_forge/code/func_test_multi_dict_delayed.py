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
def test_multi_dict_delayed(self):

    @gen.coroutine
    def f():
        responses = (yield gen.multi_future(dict(foo=self.delay(3, 'v1'), bar=self.delay(1, 'v2'))))
        self.assertEqual(responses, dict(foo='v1', bar='v2'))
    self.io_loop.run_sync(f)