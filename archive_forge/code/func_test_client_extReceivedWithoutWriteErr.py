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
def test_client_extReceivedWithoutWriteErr(self):
    """
        SSHSession.extReceived() should handle the case where the transport
        on the client doesn't have a writeErr method.
        """
    client = self.session.client = StubClient()
    client.transport = StubTransport()
    self.session.extReceived(connection.EXTENDED_DATA_STDERR, b'ignored')