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
def test_connectionCancelledBeforeSecure(self):
    """
        If the connection is cancelled before the SSH transport layer has
        finished key exchange (ie, gotten to the point where we may attempt to
        authenticate), the L{Deferred} returned by
        L{SSHCommandClientEndpoint.connect} fires with a L{Failure} wrapping
        L{CancelledError} and the connection is aborted.
        """
    endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', b'dummy user', self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
    factory = Factory()
    factory.protocol = Protocol
    d = endpoint.connect(factory)
    transport = AbortableFakeTransport(None, isServer=False)
    factory = self.reactor.tcpClients[0][2]
    client = factory.buildProtocol(None)
    client.makeConnection(transport)
    d.cancel()
    self.failureResultOf(d).trap(CancelledError)
    self.assertTrue(transport.aborted)
    client.connectionLost(Failure(ConnectionDone()))