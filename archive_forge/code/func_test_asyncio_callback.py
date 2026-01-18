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
def test_asyncio_callback(self):

    async def add_callback():
        asyncio.get_event_loop().call_soon(self.stop)
    self.asyncio_loop.run_until_complete(add_callback())
    self.wait()