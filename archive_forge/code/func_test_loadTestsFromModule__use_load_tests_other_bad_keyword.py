import functools
import sys
import types
import warnings
import unittest
@warningregistry
def test_loadTestsFromModule__use_load_tests_other_bad_keyword(self):
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with self.assertRaises(TypeError) as cm:
            loader.loadTestsFromModule(m, use_load_tests=False, very_bad=True, worse=False)
    self.assertEqual(type(cm.exception), TypeError)
    self.assertEqual(str(cm.exception), "loadTestsFromModule() got an unexpected keyword argument 'very_bad'")