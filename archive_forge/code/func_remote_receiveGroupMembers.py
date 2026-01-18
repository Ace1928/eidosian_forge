from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def remote_receiveGroupMembers(self, names, group):
    print('received group members:', names, group)
    self.getGroupConversation(group).setGroupMembers(names)