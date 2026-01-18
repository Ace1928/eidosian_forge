import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def getLastLine(self, transport):
    """
        Return the last IRC message in the transport buffer.
        """
    line = transport.value()
    if bytes != str and isinstance(line, bytes):
        line = line.decode('utf-8')
    return line.split('\r\n')[-2]