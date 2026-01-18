from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc
def imperson_whois(self):
    if self.account.client is None:
        raise locals.OfflineError
    self.account.client.sendLine('WHOIS %s' % self.name)