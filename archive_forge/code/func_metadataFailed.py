from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def metadataFailed(self, result, text):
    print('result:', result, 'text:', text)
    return self.account.client.perspective.callRemote('groupMessage', self.name, text)