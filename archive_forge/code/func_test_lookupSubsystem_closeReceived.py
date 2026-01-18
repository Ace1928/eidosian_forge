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
def test_lookupSubsystem_closeReceived(self):
    """
        SSHSession.closeReceived() should sent a close message to the remote
        side.
        """
    self.session.requestReceived(b'subsystem', common.NS(b'TestSubsystem') + b'data')
    self.session.closeReceived()
    self.assertTrue(self.session.conn.closes[self.session])