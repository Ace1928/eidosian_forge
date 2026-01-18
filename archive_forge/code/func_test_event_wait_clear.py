import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_event_wait_clear(self):
    e = locks.Event()
    f0 = asyncio.ensure_future(e.wait())
    e.clear()
    f1 = asyncio.ensure_future(e.wait())
    e.set()
    self.assertTrue(f0.done())
    self.assertTrue(f1.done())