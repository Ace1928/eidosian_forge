import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_addTest__noniterable(self):
    suite = unittest.TestSuite()
    try:
        suite.addTests(5)
    except TypeError:
        pass
    else:
        self.fail('Failed to raise TypeError')