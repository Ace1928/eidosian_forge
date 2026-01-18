import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
def test_channelOpenFailure(self):
    """
        If a channel cannot be opened on the authenticated SSH connection, the
        L{Deferred} returned by L{SSHCommandClientEndpoint.connect} fires with
        a L{Failure} wrapping the reason given by the server.
        """
    endpoint = self.create()
    factory = Factory()
    factory.protocol = Protocol
    connected = endpoint.connect(factory)
    server, client, pump = self.finishConnection()
    errors = self.flushLoggedErrors(ConchError)
    self.assertIn('unknown channel', (errors[0].value.data, errors[0].value.value))
    self.assertEqual(1, len(errors))
    f = self.failureResultOf(connected)
    f.trap(ConchError)
    self.assertEqual(b'unknown channel', f.value.value)
    self.assertClientTransportState(client, False)