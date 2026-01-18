import io
import sys
import unittest
def test_error_in_teardown_class(self):

    class Test(unittest.TestCase):
        tornDown = 0

        @classmethod
        def tearDownClass(cls):
            Test.tornDown += 1
            raise TypeError('foo')

        def test_one(self):
            pass

        def test_two(self):
            pass

    class Test2(unittest.TestCase):
        tornDown = 0

        @classmethod
        def tearDownClass(cls):
            Test2.tornDown += 1
            raise TypeError('foo')

        def test_one(self):
            pass

        def test_two(self):
            pass
    result = self.runTests(Test, Test2)
    self.assertEqual(result.testsRun, 4)
    self.assertEqual(len(result.errors), 2)
    self.assertEqual(Test.tornDown, 1)
    self.assertEqual(Test2.tornDown, 1)
    error, _ = result.errors[0]
    self.assertEqual(str(error), 'tearDownClass (%s.%s)' % (__name__, Test.__qualname__))