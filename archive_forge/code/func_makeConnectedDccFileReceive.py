import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def makeConnectedDccFileReceive(self, filename, resumeOffset=0, overwrite=None):
    """
        Factory helper that returns a L{DccFileReceive} instance
        for a specific test case.

        @param filename: Path to the local file where received data is stored.
        @type filename: L{str}

        @param resumeOffset: An integer representing the amount of bytes from
            where the transfer of data should be resumed.
        @type resumeOffset: L{int}

        @param overwrite: A boolean specifying whether the file to write to
            should be overwritten by calling L{DccFileReceive.set_overwrite}
            or not.
        @type overwrite: L{bool}

        @return: An instance of L{DccFileReceive}.
        @rtype: L{DccFileReceive}
        """
    protocol = irc.DccFileReceive(filename, resumeOffset=resumeOffset)
    if overwrite:
        protocol.set_overwrite(True)
    transport = StringTransport()
    protocol.makeConnection(transport)
    return protocol