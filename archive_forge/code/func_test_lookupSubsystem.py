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
def test_lookupSubsystem(self):
    """
        When a client requests a subsystem, the SSHSession object should get
        the subsystem by calling avatar.lookupSubsystem, and attach it as
        the client.
        """
    ret = self.session.requestReceived(b'subsystem', common.NS(b'TestSubsystem') + b'data')
    self.assertTrue(ret)
    self.assertIsInstance(self.session.client, protocol.ProcessProtocol)
    self.assertIs(self.session.client.transport.proto, self.session.avatar.subsystem)