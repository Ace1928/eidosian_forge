import unittest
from unittest.test.support import LoggingResult
def test_expected_failure_and_skip_in_cleanup(self):

    class Foo(unittest.TestCase):

        @unittest.expectedFailure
        def test_die(self):
            self.fail('help me!')

        def tearDown(self):
            self.skipTest('skip')
    events = []
    result = LoggingResult(events)
    test = Foo('test_die')
    self.assertIs(test.run(result), result)
    self.assertEqual(events, ['startTest', 'addSkip', 'stopTest'])
    self.assertFalse(result.failures)
    self.assertFalse(result.expectedFailures)
    self.assertFalse(result.unexpectedSuccesses)
    self.assertEqual(result.skipped, [(test, 'skip')])
    self.assertTrue(result.wasSuccessful())