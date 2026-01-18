import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__relative_invalid_testmethod(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('testcase_1.testfoo', m)
    expected = "type object 'MyTestCase' has no attribute 'testfoo'"
    error, test = self.check_deferred_error(loader, suite)
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected, test.testfoo)