import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_blocking_put_wait(self):
    q = queues.Queue(1)
    q.put_nowait(0)

    def get_and_discard():
        q.get()
    self.io_loop.call_later(0.01, get_and_discard)
    self.io_loop.call_later(0.02, get_and_discard)
    futures = [q.put(0), q.put(1)]
    self.assertFalse(any((f.done() for f in futures)))
    yield futures