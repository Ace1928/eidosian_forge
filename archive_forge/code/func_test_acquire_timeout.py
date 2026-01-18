import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_acquire_timeout(self):
    lock = locks.Lock()
    lock.acquire()
    with self.assertRaises(gen.TimeoutError):
        yield lock.acquire(timeout=timedelta(seconds=0.01))
    self.assertFalse(asyncio.ensure_future(lock.acquire()).done())