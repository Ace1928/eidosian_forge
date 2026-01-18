import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_addTests(self):

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass
    test_1 = Foo('test_1')
    test_2 = Foo('test_2')
    inner_suite = unittest.TestSuite([test_2])

    def gen():
        yield test_1
        yield test_2
        yield inner_suite
    suite_1 = unittest.TestSuite()
    suite_1.addTests(gen())
    self.assertEqual(list(suite_1), list(gen()))
    suite_2 = unittest.TestSuite()
    for t in gen():
        suite_2.addTest(t)
    self.assertEqual(suite_1, suite_2)