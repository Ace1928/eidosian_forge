import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_blocking_put(self):
    q = queues.Queue()
    q.put(0)
    self.assertEqual(0, q.get_nowait())