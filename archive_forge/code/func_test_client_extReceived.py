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
def test_client_extReceived(self):
    """
        SSHSession.extReceived() passed data of type EXTENDED_DATA_STDERR along
        to the client.  If the data comes before there is a client, or if the
        data is not of type EXTENDED_DATA_STDERR, it is discared.
        """
    self.session.extReceived(connection.EXTENDED_DATA_STDERR, b'1')
    self.session.extReceived(255, b'2')
    self.session.client = StubClient()
    self.session.extReceived(connection.EXTENDED_DATA_STDERR, b'3')
    self.assertEqual(self.session.client.transport.err, b'3')