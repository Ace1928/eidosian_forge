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
def test_completed_concurrent_future(self):
    with futures.ThreadPoolExecutor(1) as executor:

        def dummy():
            pass
        f = executor.submit(dummy)
        f.result()
        yield gen.with_timeout(datetime.timedelta(seconds=3600), f)