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
def test_lookupSubsystem_data(self):
    """
        After having looked up a subsystem, data should be passed along to the
        client.  Additionally, subsystems were passed the entire request packet
        as data, instead of just the additional data.

        We check for the additional tidle to verify that the data passed
        through the client.
        """
    self.session.requestReceived(b'subsystem', common.NS(b'TestSubsystem') + b'data')
    self.assertEqual(self.session.conn.data[self.session], [b'\x00\x00\x00\rTestSubsystemdata~'])
    self.session.dataReceived(b'more data')
    self.assertEqual(self.session.conn.data[self.session][-1], b'more data~')