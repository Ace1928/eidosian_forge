import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_addTests__string(self):
    suite = unittest.TestSuite()
    self.assertRaises(TypeError, suite.addTests, 'foo')