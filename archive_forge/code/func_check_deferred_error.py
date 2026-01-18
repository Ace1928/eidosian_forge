import functools
import sys
import types
import warnings
import unittest
def check_deferred_error(self, loader, suite):
    """Helper function for checking that errors in loading are reported.

        :param loader: A loader with some errors.
        :param suite: A suite that should have a late bound error.
        :return: The first error message from the loader and the test object
            from the suite.
        """
    self.assertIsInstance(suite, unittest.TestSuite)
    self.assertEqual(suite.countTestCases(), 1)
    self.assertNotEqual([], loader.errors)
    self.assertEqual(1, len(loader.errors))
    error = loader.errors[0]
    test = list(suite)[0]
    return (error, test)