import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def allDataReceivedForProtocol(self, protocol, data):
    """
        Arrange the protocol so that it received all data.

        @param protocol: The protocol which will receive the data.
        @type: L{DccFileReceive}

        @param data: The received data.
        @type data: L{bytest}
        """
    protocol.dataReceived(data)
    protocol.connectionLost(None)