import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromModule__faulty_load_tests(self):
    m = types.ModuleType('m')

    def load_tests(loader, tests, pattern):
        raise TypeError('some failure')
    m.load_tests = load_tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(m)
    self.assertIsInstance(suite, unittest.TestSuite)
    self.assertEqual(suite.countTestCases(), 1)
    self.assertNotEqual([], loader.errors)
    self.assertEqual(1, len(loader.errors))
    error = loader.errors[0]
    self.assertTrue('Failed to call load_tests:' in error, 'missing error string in %r' % error)
    test = list(suite)[0]
    self.assertRaisesRegex(TypeError, 'some failure', test.m)