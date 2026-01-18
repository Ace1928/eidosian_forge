import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_join_timeout(self):
    q = self.queue_class()
    q.put(0)
    with self.assertRaises(TimeoutError):
        yield q.join(timeout=timedelta(seconds=0.01))