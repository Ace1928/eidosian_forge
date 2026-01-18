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
@gen_test
def test_asyncio_future(self):
    x = (yield asyncio.ensure_future(asyncio.get_event_loop().run_in_executor(None, lambda: 42)))
    self.assertEqual(x, 42)