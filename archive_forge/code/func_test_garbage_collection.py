import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_garbage_collection(self):
    sem = locks.Semaphore(value=0)
    futures = [asyncio.ensure_future(sem.acquire(timedelta(seconds=0.01))) for _ in range(101)]
    future = asyncio.ensure_future(sem.acquire())
    self.assertEqual(102, len(sem._waiters))
    yield gen.sleep(0.02)
    self.assertEqual(1, len(sem._waiters))
    self.assertFalse(future.done())
    sem.release()
    self.assertTrue(future.done())
    for future in futures:
        self.assertRaises(TimeoutError, future.result)