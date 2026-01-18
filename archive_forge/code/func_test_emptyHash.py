import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_emptyHash(self):
    """
        C{__hash__} can be used to get a hash of an address, even one referring
        to L{None} rather than a real path.
        """
    addr = self.buildAddress()
    d = {addr: True}
    self.assertTrue(d[self.buildAddress()])