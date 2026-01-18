import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_differentNamesComparison(self):
    """
        Check that comparison operators work correctly on address objects
        when a different name is passed in
        """
    self.assertFalse(self.buildAddress() == self.buildDifferentAddress())
    self.assertFalse(self.buildDifferentAddress() == self.buildAddress())
    self.assertTrue(self.buildAddress() != self.buildDifferentAddress())
    self.assertTrue(self.buildDifferentAddress() != self.buildAddress())