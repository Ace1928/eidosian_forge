import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
def test_errConnectionLost(self):
    """
        Make sure reverse ordering of events in test_outConnectionLost also
        sends EOF.
        """
    self.pp.errConnectionLost()
    self.assertFalse(self.session in self.session.conn.eofs)
    self.pp.outConnectionLost()
    self.assertTrue(self.session.conn.eofs[self.session])