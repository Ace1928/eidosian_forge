import asyncio
import contextvars
import unittest
from test import support
def test_exception_in_tear_clean_up(self):

    class Test(unittest.IsolatedAsyncioTestCase):

        async def asyncSetUp(self):
            events.append('asyncSetUp')

        async def test_func(self):
            events.append('test')
            self.addAsyncCleanup(self.on_cleanup1)
            self.addAsyncCleanup(self.on_cleanup2)

        async def asyncTearDown(self):
            events.append('asyncTearDown')

        async def on_cleanup1(self):
            events.append('cleanup1')
            raise MyException('some error')

        async def on_cleanup2(self):
            events.append('cleanup2')
            raise MyException('other error')
    events = []
    test = Test('test_func')
    result = test.run()
    self.assertEqual(events, ['asyncSetUp', 'test', 'asyncTearDown', 'cleanup2', 'cleanup1'])
    self.assertIs(result.errors[0][0], test)
    self.assertIn('MyException: other error', result.errors[0][1])
    self.assertIn('MyException: some error', result.errors[1][1])
    events = []
    test = Test('test_func')
    self.addCleanup(test._tearDownAsyncioRunner)
    try:
        test.debug()
    except MyException:
        pass
    else:
        self.fail('Expected a MyException exception')
    self.assertEqual(events, ['asyncSetUp', 'test', 'asyncTearDown', 'cleanup2'])
    test.doCleanups()
    self.assertEqual(events, ['asyncSetUp', 'test', 'asyncTearDown', 'cleanup2', 'cleanup1'])