import unittest
from unittest.test.support import LoggingResult
def test_debug_skipping_class(self):

    @unittest.skip('testing')
    class Foo(unittest.TestCase):

        def setUp(self):
            events.append('setUp')

        def tearDown(self):
            events.append('tearDown')

        def test(self):
            events.append('test')
    events = []
    test = Foo('test')
    with self.assertRaises(unittest.SkipTest) as cm:
        test.debug()
    self.assertIn('testing', str(cm.exception))
    self.assertEqual(events, [])