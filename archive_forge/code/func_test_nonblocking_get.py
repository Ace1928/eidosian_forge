import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_nonblocking_get(self):
    q = queues.Queue()
    q.put_nowait(0)
    self.assertEqual(0, q.get_nowait())