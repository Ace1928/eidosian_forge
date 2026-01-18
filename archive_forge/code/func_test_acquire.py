import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_acquire(self):
    sem = locks.Semaphore()
    f0 = asyncio.ensure_future(sem.acquire())
    self.assertTrue(f0.done())
    f1 = asyncio.ensure_future(sem.acquire())
    self.assertFalse(f1.done())
    f2 = asyncio.ensure_future(sem.acquire())
    sem.release()
    self.assertTrue(f1.done())
    self.assertFalse(f2.done())
    sem.release()
    self.assertTrue(f2.done())
    sem.release()
    self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
    self.assertEqual(0, len(sem._waiters))