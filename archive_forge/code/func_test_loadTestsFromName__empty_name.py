import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__empty_name(self):
    loader = unittest.TestLoader()
    try:
        loader.loadTestsFromName('')
    except ValueError as e:
        self.assertEqual(str(e), 'Empty module name')
    else:
        self.fail('TestLoader.loadTestsFromName failed to raise ValueError')