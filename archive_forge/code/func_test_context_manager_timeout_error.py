import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_context_manager_timeout_error(self):
    sem = locks.Semaphore(value=0)
    with self.assertRaises(gen.TimeoutError):
        with (yield sem.acquire(timedelta(seconds=0.01))):
            pass
    self.assertFalse(asyncio.ensure_future(sem.acquire()).done())