import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_noNumbersVERSION(self):
    """
        If attributes for version information on L{IRCClient} are set to
        L{None}, the parts of the CTCP VERSION response they correspond to
        are omitted.
        """
    self.client.versionName = 'FrobozzIRC'
    self.client.ctcpQuery_VERSION('nick!guy@over.there', '#theChan', None)
    versionReply = 'NOTICE nick :%(X)cVERSION %(vname)s::%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF, 'vname': self.client.versionName}
    reply = self.file.getvalue()
    self.assertEqualBufferValue(reply, versionReply)