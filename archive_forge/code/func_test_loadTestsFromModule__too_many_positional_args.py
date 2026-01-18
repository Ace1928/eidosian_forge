import functools
import sys
import types
import warnings
import unittest
@warningregistry
def test_loadTestsFromModule__too_many_positional_args(self):
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
    with self.assertRaises(TypeError) as cm, warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        loader.loadTestsFromModule(m, False, 'testme.*')
    self.assertIs(w[-1].category, DeprecationWarning)
    self.assertEqual(str(w[-1].message), 'use_load_tests is deprecated and ignored')
    self.assertEqual(type(cm.exception), TypeError)
    self.assertEqual(str(cm.exception), 'loadTestsFromModule() takes 1 positional argument but 3 were given')