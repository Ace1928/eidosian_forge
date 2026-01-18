from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc
def irc_RPL_ENDOFNAMES(self, prefix, params):
    group = params[1][1:]
    self.getGroupConversation(group).setGroupMembers(self._namreplies[group.lower()])
    del self._namreplies[group.lower()]