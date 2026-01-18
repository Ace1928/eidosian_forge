import asyncio
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from tornado.ioloop import IOLoop
from tornado.platform.asyncio import (
from tornado.testing import AsyncTestCase, gen_test
def test_asyncio_accessor(self):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        self.assertRaises(RuntimeError, self.executor.submit(asyncio.get_event_loop).result)
        asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        self.assertIsInstance(self.executor.submit(asyncio.get_event_loop).result(), asyncio.AbstractEventLoop)
        self.executor.submit(lambda: asyncio.get_event_loop().close()).result()