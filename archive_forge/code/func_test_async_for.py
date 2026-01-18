import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_async_for(self):
    q = queues.Queue()
    for i in range(5):
        q.put(i)

    async def f():
        results = []
        async for i in q:
            results.append(i)
            if i == 4:
                return results
    results = (yield f())
    self.assertEqual(results, list(range(5)))