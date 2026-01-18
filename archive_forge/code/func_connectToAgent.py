import signal
from os.path import expanduser
from struct import unpack
from zope.interface import Interface, implementer
from twisted.conch.client.agent import SSHAgentClient
from twisted.conch.client.default import _KNOWN_HOSTS
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.ssh.channel import SSHChannel
from twisted.conch.ssh.common import NS, getNS
from twisted.conch.ssh.connection import SSHConnection
from twisted.conch.ssh.keys import Key
from twisted.conch.ssh.transport import SSHClientTransport
from twisted.conch.ssh.userauth import SSHUserAuthClient
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.error import ConnectionDone, ProcessTerminated
from twisted.internet.interfaces import IStreamClientEndpoint
from twisted.internet.protocol import Factory
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
def connectToAgent(self, endpoint):
    """
        Set up a connection to the authentication agent and trigger its
        initialization.

        @param endpoint: An endpoint which can be used to connect to the
            authentication agent.
        @type endpoint: L{IStreamClientEndpoint} provider

        @return: A L{Deferred} which fires when the agent connection is ready
            for use.
        """
    factory = Factory()
    factory.protocol = SSHAgentClient
    d = endpoint.connect(factory)

    def connected(agent):
        self.agent = agent
        return agent.getPublicKeys()
    d.addCallback(connected)
    return d