import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_nonblocking_put_with_getters(self):
    q = queues.Queue()
    get0 = q.get()
    get1 = q.get()
    q.put_nowait(0)
    yield gen.moment
    self.assertEqual(0, (yield get0))
    q.put_nowait(1)
    yield gen.moment
    self.assertEqual(1, (yield get1))