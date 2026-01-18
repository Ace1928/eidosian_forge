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
def test_ioloop_close_leak(self):
    orig_count = len(IOLoop._ioloop_for_asyncio)
    for i in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            loop = AsyncIOLoop()
            loop.close()
    new_count = len(IOLoop._ioloop_for_asyncio) - orig_count
    self.assertEqual(new_count, 0)