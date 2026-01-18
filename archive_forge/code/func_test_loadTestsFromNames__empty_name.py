import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__empty_name(self):
    loader = unittest.TestLoader()
    try:
        loader.loadTestsFromNames([''])
    except ValueError as e:
        self.assertEqual(str(e), 'Empty module name')
    else:
        self.fail('TestLoader.loadTestsFromNames failed to raise ValueError')