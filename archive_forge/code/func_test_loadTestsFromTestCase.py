import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromTestCase(self):

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass

        def foo_bar(self):
            pass
    tests = unittest.TestSuite([Foo('test_1'), Foo('test_2')])
    loader = unittest.TestLoader()
    self.assertEqual(loader.loadTestsFromTestCase(Foo), tests)