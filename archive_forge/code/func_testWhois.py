import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testWhois(self):
    """
        Verify that a whois by the client receives the right protocol actions
        from the server.
        """
    timestamp = int(time.time() - 100)
    hostname = self.p.hostname
    req = 'requesting-nick'
    targ = 'target-nick'
    self.p.whois(req, targ, 'target', 'host.com', 'Target User', 'irc.host.com', 'A fake server', False, 12, timestamp, ['#fakeusers', '#fakemisc'])
    lines = [':%(hostname)s 311 %(req)s %(targ)s target host.com * :Target User', ':%(hostname)s 312 %(req)s %(targ)s irc.host.com :A fake server', ':%(hostname)s 317 %(req)s %(targ)s 12 %(timestamp)s :seconds idle, signon time', ':%(hostname)s 319 %(req)s %(targ)s :#fakeusers #fakemisc', ':%(hostname)s 318 %(req)s %(targ)s :End of WHOIS list.', '']
    expected = '\r\n'.join(lines) % dict(hostname=hostname, timestamp=timestamp, req=req, targ=targ)
    self.check(expected)