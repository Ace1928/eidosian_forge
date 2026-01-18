import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_nested_notify(self):
    c = locks.Condition()
    futures = [asyncio.ensure_future(c.wait()) for _ in range(3)]
    futures[1].add_done_callback(lambda _: c.notify())
    c.notify(2)
    yield
    self.assertTrue(all((f.done() for f in futures)))