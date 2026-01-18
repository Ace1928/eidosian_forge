import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_get_clears_timed_out_getters(self):
    q = queues.Queue()
    getters = [asyncio.ensure_future(q.get(timedelta(seconds=0.01))) for _ in range(10)]
    get = asyncio.ensure_future(q.get())
    self.assertEqual(11, len(q._getters))
    yield gen.sleep(0.02)
    self.assertEqual(11, len(q._getters))
    self.assertFalse(get.done())
    q.get()
    self.assertEqual(2, len(q._getters))
    for getter in getters:
        self.assertRaises(TimeoutError, getter.result)