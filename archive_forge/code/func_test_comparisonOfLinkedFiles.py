import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
@skipIf(symlinkSkip, 'Platform does not support symlinks')
def test_comparisonOfLinkedFiles(self):
    """
        A UNIXAddress referring to a L{None} address does not
        compare equal to a UNIXAddress referring to a symlink.
        """
    linkName = self.mktemp()
    with open(self._socketAddress, 'w') as self.fd:
        os.symlink(os.path.abspath(self._socketAddress), linkName)
        self.assertNotEqual(UNIXAddress(self._socketAddress), UNIXAddress(None))
        self.assertNotEqual(UNIXAddress(None), UNIXAddress(self._socketAddress))