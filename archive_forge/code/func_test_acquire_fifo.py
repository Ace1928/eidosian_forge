import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_acquire_fifo(self):
    lock = locks.Lock()
    self.assertTrue(asyncio.ensure_future(lock.acquire()).done())
    N = 5
    history = []

    @gen.coroutine
    def f(idx):
        with (yield lock.acquire()):
            history.append(idx)
    futures = [f(i) for i in range(N)]
    self.assertFalse(any((future.done() for future in futures)))
    lock.release()
    yield futures
    self.assertEqual(list(range(N)), history)