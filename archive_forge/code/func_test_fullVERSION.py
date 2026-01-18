import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_fullVERSION(self):
    """
        The response to a CTCP VERSION query includes the version number and
        environment information, as specified by L{IRCClient.versionNum} and
        L{IRCClient.versionEnv}.
        """
    self.client.versionName = 'FrobozzIRC'
    self.client.versionNum = '1.2g'
    self.client.versionEnv = 'ZorkOS'
    self.client.ctcpQuery_VERSION('nick!guy@over.there', '#theChan', None)
    versionReply = 'NOTICE nick :%(X)cVERSION %(vname)s:%(vnum)s:%(venv)s%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF, 'vname': self.client.versionName, 'vnum': self.client.versionNum, 'venv': self.client.versionEnv}
    reply = self.file.getvalue()
    self.assertEqualBufferValue(reply, versionReply)