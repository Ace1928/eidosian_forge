import asyncio
import contextvars
import unittest
from test import support
def test_full_cycle(self):
    expected = ['setUp', 'asyncSetUp', 'test', 'asyncTearDown', 'tearDown', 'cleanup6', 'cleanup5', 'cleanup4', 'cleanup3', 'cleanup2', 'cleanup1']

    class Test(unittest.IsolatedAsyncioTestCase):

        def setUp(self):
            self.assertEqual(events, [])
            events.append('setUp')
            VAR.set(VAR.get() + ('setUp',))
            self.addCleanup(self.on_cleanup1)
            self.addAsyncCleanup(self.on_cleanup2)

        async def asyncSetUp(self):
            self.assertEqual(events, expected[:1])
            events.append('asyncSetUp')
            VAR.set(VAR.get() + ('asyncSetUp',))
            self.addCleanup(self.on_cleanup3)
            self.addAsyncCleanup(self.on_cleanup4)

        async def test_func(self):
            self.assertEqual(events, expected[:2])
            events.append('test')
            VAR.set(VAR.get() + ('test',))
            self.addCleanup(self.on_cleanup5)
            self.addAsyncCleanup(self.on_cleanup6)

        async def asyncTearDown(self):
            self.assertEqual(events, expected[:3])
            VAR.set(VAR.get() + ('asyncTearDown',))
            events.append('asyncTearDown')

        def tearDown(self):
            self.assertEqual(events, expected[:4])
            events.append('tearDown')
            VAR.set(VAR.get() + ('tearDown',))

        def on_cleanup1(self):
            self.assertEqual(events, expected[:10])
            events.append('cleanup1')
            VAR.set(VAR.get() + ('cleanup1',))
            nonlocal cvar
            cvar = VAR.get()

        async def on_cleanup2(self):
            self.assertEqual(events, expected[:9])
            events.append('cleanup2')
            VAR.set(VAR.get() + ('cleanup2',))

        def on_cleanup3(self):
            self.assertEqual(events, expected[:8])
            events.append('cleanup3')
            VAR.set(VAR.get() + ('cleanup3',))

        async def on_cleanup4(self):
            self.assertEqual(events, expected[:7])
            events.append('cleanup4')
            VAR.set(VAR.get() + ('cleanup4',))

        def on_cleanup5(self):
            self.assertEqual(events, expected[:6])
            events.append('cleanup5')
            VAR.set(VAR.get() + ('cleanup5',))

        async def on_cleanup6(self):
            self.assertEqual(events, expected[:5])
            events.append('cleanup6')
            VAR.set(VAR.get() + ('cleanup6',))
    events = []
    cvar = ()
    test = Test('test_func')
    result = test.run()
    self.assertEqual(result.errors, [])
    self.assertEqual(result.failures, [])
    self.assertEqual(events, expected)
    self.assertEqual(cvar, tuple(expected))
    events = []
    cvar = ()
    test = Test('test_func')
    test.debug()
    self.assertEqual(events, expected)
    self.assertEqual(cvar, tuple(expected))
    test.doCleanups()
    self.assertEqual(events, expected)
    self.assertEqual(cvar, tuple(expected))