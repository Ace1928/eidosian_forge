import unittest
from unittest.test.support import LoggingResult
def test_debug_skipping(self):

    class Foo(unittest.TestCase):

        def setUp(self):
            events.append('setUp')

        def tearDown(self):
            events.append('tearDown')

        def test1(self):
            self.skipTest('skipping exception')
            events.append('test1')

        @unittest.skip('skipping decorator')
        def test2(self):
            events.append('test2')
    events = []
    test = Foo('test1')
    with self.assertRaises(unittest.SkipTest) as cm:
        test.debug()
    self.assertIn('skipping exception', str(cm.exception))
    self.assertEqual(events, ['setUp'])
    events = []
    test = Foo('test2')
    with self.assertRaises(unittest.SkipTest) as cm:
        test.debug()
    self.assertIn('skipping decorator', str(cm.exception))
    self.assertEqual(events, [])