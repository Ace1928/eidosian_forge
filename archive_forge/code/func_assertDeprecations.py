import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def assertDeprecations(self, testMethod, message):
    """
        Assert that the a DeprecationWarning with the given message was
        emitted against the given method.
        """
    warnings = self.flushWarnings([testMethod])
    self.assertEqual(warnings[0]['category'], DeprecationWarning)
    self.assertEqual(warnings[0]['message'], message)
    self.assertEqual(len(warnings), 1)