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
def test_asyncio_adapter(self):

    @gen.coroutine
    def tornado_coroutine():
        yield gen.moment
        raise gen.Return(42)

    async def native_coroutine_without_adapter():
        return await tornado_coroutine()

    async def native_coroutine_with_adapter():
        return await to_asyncio_future(tornado_coroutine())

    async def native_coroutine_with_adapter2():
        return await to_asyncio_future(native_coroutine_without_adapter())
    self.assertEqual(self.io_loop.run_sync(native_coroutine_without_adapter), 42)
    self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter), 42)
    self.assertEqual(self.io_loop.run_sync(native_coroutine_with_adapter2), 42)
    self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_without_adapter()), 42)
    self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_with_adapter()), 42)
    self.assertEqual(self.asyncio_loop.run_until_complete(native_coroutine_with_adapter2()), 42)