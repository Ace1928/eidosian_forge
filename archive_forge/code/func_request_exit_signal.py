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
def request_exit_signal(self, data):
    """
        When the server sends the command's exit status, record it for later
        delivery to the protocol.

        @param data: The network-order four byte representation of the exit
            signal of the command.
        @type data: L{bytes}
        """
    shortSignalName, data = getNS(data)
    coreDumped, data = (bool(ord(data[0:1])), data[1:])
    errorMessage, data = getNS(data)
    languageTag, data = getNS(data)
    signalName = f'SIG{nativeString(shortSignalName)}'
    signalID = getattr(signal, signalName, -1)
    self._log.info('Process exited with signal {shortSignalName!r}; core dumped: {coreDumped}; error message: {errorMessage}; language: {languageTag!r}', shortSignalName=shortSignalName, coreDumped=coreDumped, errorMessage=errorMessage.decode('utf-8'), languageTag=languageTag)
    self._reason = ProcessTerminated(None, signalID, None)