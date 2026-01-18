import io
import sys
import unittest
def test_error_in_setupclass(self):

    class BrokenTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            raise TypeError('foo')

        def test_one(self):
            pass

        def test_two(self):
            pass
    result = self.runTests(BrokenTest)
    self.assertEqual(result.testsRun, 0)
    self.assertEqual(len(result.errors), 1)
    error, _ = result.errors[0]
    self.assertEqual(str(error), 'setUpClass (%s.%s)' % (__name__, BrokenTest.__qualname__))