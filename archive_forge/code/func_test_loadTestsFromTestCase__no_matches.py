import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromTestCase__no_matches(self):

    class Foo(unittest.TestCase):

        def foo_bar(self):
            pass
    empty_suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    self.assertEqual(loader.loadTestsFromTestCase(Foo), empty_suite)