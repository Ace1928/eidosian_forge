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
def test_multi_moment(self):

    @gen.coroutine
    def wait_a_moment():
        result = (yield gen.multi([gen.moment, gen.moment]))
        raise gen.Return(result)
    loop = self.get_new_ioloop()
    result = loop.run_sync(wait_a_moment)
    self.assertEqual(result, [None, None])