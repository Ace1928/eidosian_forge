from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def registerMany(results):
    for success, result in results:
        if success:
            chatui.registerAccountClient(result)
            self._cb_logOn(result)
        else:
            log.err(result)