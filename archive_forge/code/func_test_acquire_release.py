import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_acquire_release(self):
    lock = locks.Lock()
    self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
    future = asyncio.ensure_future(lock.acquire())
    self.assertFalse(future.done())
    lock.release()
    self.assertTrue(future.done())