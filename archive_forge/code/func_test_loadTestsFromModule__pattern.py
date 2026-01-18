import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromModule__pattern(self):
    m = types.ModuleType('m')

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass
    m.testcase_1 = MyTestCase
    load_tests_args = []

    def load_tests(loader, tests, pattern):
        self.assertIsInstance(tests, unittest.TestSuite)
        load_tests_args.extend((loader, tests, pattern))
        return tests
    m.load_tests = load_tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(m, pattern='testme.*')
    self.assertIsInstance(suite, unittest.TestSuite)
    self.assertEqual(load_tests_args, [loader, suite, 'testme.*'])