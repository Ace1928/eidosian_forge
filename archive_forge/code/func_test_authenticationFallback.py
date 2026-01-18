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
def test_authenticationFallback(self):
    """
        If the SSH server does not accept any of the specified SSH keys, the
        specified password is tried.
        """
    badKey = Key.fromString(privateRSA_openssh)
    self.setupKeyChecker(self.portal, {self.user: privateDSA_openssh})
    endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, keys=[badKey], password=self.password, knownHosts=self.knownHosts, ui=FixedResponseUI(False))
    factory = Factory()
    factory.protocol = Protocol
    connected = endpoint.connect(factory)
    self.factory.attemptsBeforeDisconnect += 1
    server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
    pump.pump()
    errors = self.flushLoggedErrors(ConchError)
    self.assertIn('unknown channel', (errors[0].value.data, errors[0].value.value))
    self.assertEqual(1, len(errors))
    f = self.failureResultOf(connected)
    f.trap(ConchError)
    self.assertEqual(b'unknown channel', f.value.value)
    self.assertTrue(client.transport.disconnecting)