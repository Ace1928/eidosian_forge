import asyncio
import contextvars
import unittest
from test import support
def test_enterAsyncContext_arg_errors(self):

    class Test(unittest.IsolatedAsyncioTestCase):

        async def test_func(slf):
            with self.assertRaisesRegex(TypeError, 'asynchronous context manager'):
                await slf.enterAsyncContext(LacksEnterAndExit())
            with self.assertRaisesRegex(TypeError, 'asynchronous context manager'):
                await slf.enterAsyncContext(LacksEnter())
            with self.assertRaisesRegex(TypeError, 'asynchronous context manager'):
                await slf.enterAsyncContext(LacksExit())
    test = Test('test_func')
    output = test.run()
    self.assertTrue(output.wasSuccessful())