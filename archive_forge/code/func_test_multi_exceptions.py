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
def test_multi_exceptions(self):
    with ExpectLog(app_log, 'Multiple exceptions in yield list'):
        with self.assertRaises(RuntimeError) as cm:
            yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))])
    self.assertEqual(str(cm.exception), 'error 1')
    with self.assertRaises(RuntimeError):
        yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_future(2)])
    with self.assertRaises(RuntimeError):
        yield gen.Multi([self.async_exception(RuntimeError('error 1')), self.async_exception(RuntimeError('error 2'))], quiet_exceptions=RuntimeError)