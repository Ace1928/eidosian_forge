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
def test_agentAuthentication(self):
    """
        If L{SSHCommandClientEndpoint} is initialized with an
        L{SSHAgentClient}, the agent is used to authenticate with the SSH
        server. Once the connection with the SSH server has concluded, the
        connection to the agent is disconnected.
        """
    key = Key.fromString(privateRSA_openssh)
    agentServer = SSHAgentServer()
    agentServer.factory = Factory()
    agentServer.factory.keys = {key.blob(): (key, b'')}
    self.setupKeyChecker(self.portal, {self.user: privateRSA_openssh})
    agentEndpoint = SingleUseMemoryEndpoint(agentServer)
    endpoint = SSHCommandClientEndpoint.newConnection(self.reactor, b'/bin/ls -l', self.user, self.hostname, self.port, knownHosts=self.knownHosts, ui=FixedResponseUI(False), agentEndpoint=agentEndpoint)
    self.realm.channelLookup[b'session'] = WorkingExecSession
    factory = Factory()
    factory.protocol = Protocol
    connected = endpoint.connect(factory)
    server, client, pump = self.connectedServerAndClient(self.factory, self.reactor.tcpClients[0][2])
    for i in range(14):
        agentEndpoint.pump.pump()
        pump.pump()
    protocol = self.successResultOf(connected)
    self.assertIsNotNone(protocol.transport)
    self.loseConnectionToServer(server, client, protocol, pump)
    self.assertTrue(client.transport.disconnecting)
    self.assertTrue(agentEndpoint.pump.clientIO.disconnecting)