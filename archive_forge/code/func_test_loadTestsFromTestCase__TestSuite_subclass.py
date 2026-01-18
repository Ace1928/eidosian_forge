import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromTestCase__TestSuite_subclass(self):

    class NotATestCase(unittest.TestSuite):
        pass
    loader = unittest.TestLoader()
    try:
        loader.loadTestsFromTestCase(NotATestCase)
    except TypeError:
        pass
    else:
        self.fail('Should raise TypeError')