import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_get_with_putters(self):
    q = queues.Queue(1)
    q.put_nowait(0)
    put = q.put(1)
    self.assertEqual(0, (yield q.get()))
    self.assertIsNone((yield put))