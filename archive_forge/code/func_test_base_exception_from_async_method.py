import asyncio
import contextvars
import unittest
from test import support
def test_base_exception_from_async_method(self):
    events = []

    class Test(unittest.IsolatedAsyncioTestCase):

        async def test_base(self):
            events.append('test_base')
            raise BaseException()
            events.append('not it')

        async def test_no_err(self):
            events.append('test_no_err')

        async def test_cancel(self):
            raise asyncio.CancelledError()
    test = Test('test_base')
    output = test.run()
    self.assertFalse(output.wasSuccessful())
    test = Test('test_no_err')
    test.run()
    self.assertEqual(events, ['test_base', 'test_no_err'])
    test = Test('test_cancel')
    output = test.run()
    self.assertFalse(output.wasSuccessful())