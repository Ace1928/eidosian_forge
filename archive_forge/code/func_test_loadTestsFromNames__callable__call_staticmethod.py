import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__callable__call_staticmethod(self):
    m = types.ModuleType('m')

    class Test1(unittest.TestCase):

        def test(self):
            pass
    testcase_1 = Test1('test')

    class Foo(unittest.TestCase):

        @staticmethod
        def foo():
            return testcase_1
    m.Foo = Foo
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['Foo.foo'], m)
    self.assertIsInstance(suite, loader.suiteClass)
    ref_suite = unittest.TestSuite([testcase_1])
    self.assertEqual(list(suite), [ref_suite])