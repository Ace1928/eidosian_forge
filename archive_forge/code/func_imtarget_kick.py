from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc
def imtarget_kick(self, target):
    if self.account.client is None:
        raise locals.OfflineError
    reason = 'for great justice!'
    self.account.client.sendLine(f'KICK #{self.name} {target.name} :{reason}')