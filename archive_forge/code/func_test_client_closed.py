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
def test_client_closed(self):
    """
        SSHSession.closed() should tell the transport connected to the client
        that the connection was lost.
        """
    self.session.client = StubClient()
    self.session.closed()
    self.assertTrue(self.session.client.transport.close)
    self.session.client.transport.close = False