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
def test_multi_future(self):
    results = (yield [self.async_future(1), self.async_future(2)])
    self.assertEqual(results, [1, 2])